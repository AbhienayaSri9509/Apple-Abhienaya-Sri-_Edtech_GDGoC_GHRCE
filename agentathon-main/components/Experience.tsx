import React from 'react';
import { Canvas } from '@react-three/fiber';
import { Environment, ContactShadows, Stars, OrbitControls } from '@react-three/drei';
import Avatar from './Avatar';
import { AvatarState } from '../types';

// Augment JSX namespace to recognize Three.js elements
declare global {
  namespace JSX {
    interface IntrinsicElements {
      ambientLight: any;
      spotLight: any;
      pointLight: any;
      group: any;
    }
  }
}

interface ExperienceProps {
  avatarState: AvatarState;
}

const Experience: React.FC<ExperienceProps> = ({ avatarState }) => {
  return (
    <Canvas
      camera={{ position: [0, 1, 5], fov: 45 }}
      shadows
      className="w-full h-full"
    >
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <spotLight 
        position={[10, 10, 10]} 
        angle={0.15} 
        penumbra={1} 
        intensity={1} 
        castShadow 
      />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#db2777" />

      {/* Camera Controls - Enable zoom, rotate, and pan */}
      <OrbitControls
        enableZoom={true}
        enableRotate={true}
        enablePan={true}
        minDistance={1}
        maxDistance={10}
        minPolarAngle={0}
        maxPolarAngle={Math.PI / 1.5}
        target={[0, 1, 0]}
      />

      {/* Environment */}
      <Environment preset="city" />
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
      
      {/* The Avatar */}
      <group position={[0, -0.5, 0]}>
        <Avatar state={avatarState} />
      </group>

      {/* Shadows */}
      <ContactShadows 
        resolution={1024} 
        scale={10} 
        blur={2} 
        opacity={0.5} 
        far={10} 
        color="#831843" 
      />
    </Canvas>
  );
};

export default Experience;