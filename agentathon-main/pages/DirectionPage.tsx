import React, { useEffect, useRef, useState } from 'react';
import Layout from '../components/Layout';

// Prefer env var the rest of the app already uses
const GOOGLE_MAPS_API_KEY =
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_GOOGLE_MAPS_API_KEY) ||
    (typeof process !== 'undefined' && process.env?.VITE_GOOGLE_MAPS_API_KEY) ||
    '';

const DirectionPage: React.FC = () => {
    const mapRef = useRef<HTMLDivElement | null>(null);
    // Only store user-facing error messages (no dev/debug text)
    const [mapError, setMapError] = useState<string>('');
    const [destination, setDestination] = useState<string>('');
    const [isRouting, setIsRouting] = useState<boolean>(false);
    const mapInstanceRef = useRef<any>(null);
    const markerRef = useRef<any>(null);
    const pendingLocationRef = useRef<{ lat: number; lng: number } | null>(null);
    const originRef = useRef<{ lat: number; lng: number } | null>(null);
    const directionsServiceRef = useRef<any>(null);
    const directionsRendererRef = useRef<any>(null);

    useEffect(() => {
        if (!GOOGLE_MAPS_API_KEY) {
            console.error('Missing VITE_GOOGLE_MAPS_API_KEY');
            setMapError('Map configuration error. Please contact support.');
            return;
        }

        // If script is already on the page, reuse it
        const existingScript = document.querySelector<HTMLScriptElement>('script[data-map-loader="google"]');
        if (existingScript && (window as any).google) {
            initMap();
            return;
        }

        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${GOOGLE_MAPS_API_KEY}`;
        script.async = true;
        script.defer = true;
        script.dataset.mapLoader = 'google';
        script.onload = initMap;
        script.onerror = () => {
            console.error('Failed to load Google Maps script');
            setMapError('Unable to load the map right now. Please check your connection and try again.');
        };
        document.head.appendChild(script);

        return () => {
            // Keep script cached; just remove map instance on unmount
        };
    }, []);

    const initMap = () => {
        if (!mapRef.current || !(window as any).google?.maps) return;

        // Default center (Chennai) until we get real device location
        const defaultLocation = { lat: 13.0827, lng: 80.2707 };

        const map = new (window as any).google.maps.Map(mapRef.current, {
            zoom: 12,
            center: defaultLocation,
        });

        mapInstanceRef.current = map;
        directionsServiceRef.current = new (window as any).google.maps.DirectionsService();
        directionsRendererRef.current = new (window as any).google.maps.DirectionsRenderer({
            map,
            suppressMarkers: false,
        });

        // If we already have a pending device location, apply it now
        if (pendingLocationRef.current) {
            const loc = pendingLocationRef.current;
            map.setCenter(loc);
            map.setZoom(15);
            markerRef.current = new (window as any).google.maps.Marker({
                position: loc,
                map,
            });
            pendingLocationRef.current = null;
        }

        // No UI message needed when map loads successfully
    };

    // Request device location as soon as the page loads
    useEffect(() => {
        if (!('geolocation' in navigator)) {
            setMapError('Location is not supported on this device/browser.');
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const loc = {
                    lat: position.coords.latitude,
                    lng: position.coords.longitude,
                };

                // Store origin for routing
                originRef.current = loc;

                // If map is ready, center it on the device; otherwise store for later
                if ((window as any).google?.maps && mapInstanceRef.current) {
                    mapInstanceRef.current.setCenter(loc);
                    mapInstanceRef.current.setZoom(15);

                    if (markerRef.current) {
                        markerRef.current.setMap(null);
                    }
                    markerRef.current = new (window as any).google.maps.Marker({
                        position: loc,
                        map: mapInstanceRef.current,
                    });
                } else {
                    pendingLocationRef.current = loc;
                }

                // No UI message needed when location works
            },
            (error) => {
                console.error('Geolocation error:', error);
                if (error.code === 1) {
                    setMapError('Location permission denied. Please enable it to center the map on your position.');
                } else if (error.code === 2) {
                    setMapError('Location unavailable. Check your GPS or network and try again.');
                } else if (error.code === 3) {
                    setMapError('Getting your location timed out. Please try again.');
                } else {
                    setMapError('Unable to get your location.');
                }
            },
            { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
        );
    }, []);

    // Geocode destination first, then request directions (more reliable errors)
    const handleGetDirections = () => {
        if (!destination.trim()) {
            setMapError('Please enter a destination.');
            return;
        }
        setMapError('');

        const googleAny = (window as any);
        if (!googleAny.google?.maps || !directionsServiceRef.current || !directionsRendererRef.current) {
            setMapError('Map is not ready yet. Please wait a moment and try again.');
            return;
        }

        const origin = originRef.current || { lat: 13.0827, lng: 80.2707 };
        const destText = destination.trim();

        setIsRouting(true);

        // Step 1: Geocode destination to catch NOT_FOUND early
        const geocoder = new googleAny.google.maps.Geocoder();
        geocoder.geocode({ address: destText }, (geoResults: any, geoStatus: string) => {
            if (geoStatus !== 'OK' || !geoResults || !geoResults[0]) {
                setIsRouting(false);
                console.error('Geocode failed:', geoStatus, geoResults);
                if (geoStatus === 'ZERO_RESULTS' || geoStatus === 'NOT_FOUND') {
                    setMapError('Destination not found. Try a full address or nearby landmark.');
                } else {
                    setMapError(`Unable to find that place (${geoStatus}). Please try again.`);
                }
                return;
            }

            const destinationAddress = geoResults[0].formatted_address;

            // Step 2: Request directions (use driving to be more permissive)
            directionsServiceRef.current.route(
                {
                    origin,
                    destination: destinationAddress,
                    travelMode: googleAny.google.maps.TravelMode.DRIVING,
                    provideRouteAlternatives: true,
                },
                (result: any, status: string) => {
                    setIsRouting(false);
                    if (status === 'OK' && result) {
                        directionsRendererRef.current.setDirections(result);
                        // Clear previous marker if any, Directions API will show markers
                        if (markerRef.current) {
                            markerRef.current.setMap(null);
                            markerRef.current = null;
                        }
                    } else {
                        console.error('Directions request failed with status:', status, result);
                        if (status === 'ZERO_RESULTS') {
                            setMapError('No route found. Try a nearby landmark or different mode.');
                        } else if (status === 'NOT_FOUND') {
                            setMapError('Destination not found. Try a full address or well-known place.');
                        } else if (status === 'REQUEST_DENIED') {
                            setMapError('Request denied. Check API key referrer and billing settings.');
                        } else if (status === 'OVER_QUERY_LIMIT') {
                            setMapError('Query limit reached. Try again later.');
                        } else {
                            setMapError(`Unable to calculate route. (${status}) Please try again with a nearby landmark or full address.`);
                        }
                    }
                }
            );
        });
    };

    return (
        <Layout>
            <div className="w-full h-screen p-4 flex flex-col gap-4">
                <h1 className="text-2xl font-semibold text-white">Directions</h1>

                <div className="flex flex-col md:flex-row gap-2 items-stretch md:items-center">
                    <input
                        type="text"
                        value={destination}
                        onChange={(e) => setDestination(e.target.value)}
                        placeholder="Enter your destination (address or place)"
                        className="flex-1 rounded-lg bg-black/40 border border-white/10 px-3 py-2 text-sm text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                    <button
                        onClick={handleGetDirections}
                        disabled={isRouting}
                        className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 disabled:cursor-not-allowed text-sm font-medium text-white transition-colors"
                    >
                        {isRouting ? 'Finding route…' : 'Get Directions'}
                    </button>
                </div>
                <div
                    ref={mapRef}
                    id="map"
                    className="w-full h-[70vh] min-h-[400px] rounded-xl overflow-hidden"
                />
                {mapError && (
                    <p className="mt-4 text-sm text-red-200 bg-red-900/40 border border-red-500/30 rounded-md px-3 py-2">
                        {mapError}
                    </p>
                )}
            </div>
        </Layout>
    );
};

export default DirectionPage;
