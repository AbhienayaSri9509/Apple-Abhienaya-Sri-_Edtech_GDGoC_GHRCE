import React, { useRef, useState, useEffect, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sparkles, useGLTF, useAnimations } from '@react-three/drei';
import * as THREE from 'three';
import { AvatarState } from '../types';

interface AvatarProps {
  state: AvatarState;
}

const Avatar: React.FC<AvatarProps> = ({ state }) => {
  const groupRef = useRef<THREE.Group>(null);
  const modelRef = useRef<THREE.Object3D>(null);

  // Load the GLB model
  const { scene } = useGLTF('/64f1a714fe61576b46f27ca2.glb');

  // Load animations
  const { animations } = useGLTF('/animations.glb');

  // Bind animations directly to the model
  const { actions, mixer } = useAnimations(animations, modelRef);

  // precise animation selection
  const animationsByName = useMemo(() => {
    const map: Record<string, THREE.AnimationClip | null> = {
      idle: null,
      talking: null,
      thinking: null
    };

    if (animations.length > 0) {
      // Find Idle
      map.idle = animations.find(a => a.name === "Idle") || animations[0];

      // Find Talking
      map.talking = animations.find(a => a.name === "Talking") ||
        animations.find(a => a.name.toLowerCase().includes('talk')) ||
        map.idle;

      // Find Thinking (or use Idle if not found)
      map.thinking = animations.find(a => a.name === "Thinking") ||
        animations.find(a => a.name.toLowerCase().includes('think')) ||
        map.idle;
    }
    return map;
  }, [animations]);

  const idleAnimation = animationsByName.idle;
  const talkingAnimation = animationsByName.talking;
  const thinkingAnimation = animationsByName.thinking;

  // Debug: Log available animations (remove in production)
  useEffect(() => {
    if (animations.length > 0) {
      console.log('Available animations:', animations.map(a => a.name));
      console.log('Mapped animations:', {
        idle: idleAnimation?.name,
        talking: talkingAnimation?.name,
        thinking: thinkingAnimation?.name
      });
    }
  }, [animations, idleAnimation, talkingAnimation, thinkingAnimation]);

  // Current animation state
  const [currentAnimation, setCurrentAnimation] = useState<string | null>(null);

  // Ensure something is always playing to avoid T-pose on idle/enter
  useEffect(() => {
    if (!actions || animations.length === 0) return;
    const defaultName = idleAnimation?.name || animations[0]?.name;
    if (defaultName && actions[defaultName] && !actions[defaultName]!.isRunning()) {
      setCurrentAnimation(defaultName);
      actions[defaultName]!.reset().fadeIn(0.2).setLoop(THREE.LoopRepeat, Infinity).play();
    }
  }, [actions, animations, idleAnimation]);

  // Handle animation changes - Always play idle when static, play specific animations when talking/thinking
  useEffect(() => {
    if (!actions || animations.length === 0 || !scene) {
      return;
    }

    // Determine which animation to play
    let targetAnimation = idleAnimation; // Default to idle

    if (state.isThinking && thinkingAnimation) {
      targetAnimation = thinkingAnimation;
    } else if (state.isTalking && talkingAnimation) {
      targetAnimation = talkingAnimation;
    }

    // Stop all animations first
    Object.values(actions).forEach((action) => {
      if (action && action.isRunning() && action !== actions[targetAnimation?.name || '']) {
        action.fadeOut(0.3);
      }
    });

    // Play the target animation (fallback to first available)
    const targetName = targetAnimation?.name || Object.keys(actions)[0];
    const action = targetName ? actions[targetName] : undefined;
    if (action && !action.isRunning()) {
      setCurrentAnimation(targetName);
      action.reset().fadeIn(0.3).setLoop(THREE.LoopRepeat, Infinity).play();
    }

    // If nothing is running, force the first available clip
    const anyRunning = Object.values(actions).some(a => a?.isRunning && a.isRunning());
    if (!anyRunning) {
      const firstName = Object.keys(actions)[0];
      if (firstName && actions[firstName]) {
        actions[firstName].reset().fadeIn(0.2).setLoop(THREE.LoopRepeat, Infinity).play();
        setCurrentAnimation(firstName);
      }
    }

    return () => {
      // Cleanup on unmount
      Object.values(actions).forEach((action) => {
        if (action && action.isRunning()) {
          action.fadeOut(0.3);
        }
      });
    };
  }, [state.isTalking, state.isThinking, actions, talkingAnimation, thinkingAnimation, idleAnimation, animations.length, scene]);

  // Remove any emissive color effects - keep materials natural
  useEffect(() => {
    if (!scene) return;

    scene.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material) {
        const material = child.material as THREE.MeshStandardMaterial;
        if (material.isMeshStandardMaterial) {
          material.emissive = new THREE.Color(0x000000);
          material.emissiveIntensity = 0;
        }
      }
    });
  }, [scene]);

  // Update mixer in useFrame to ensure animations play
  useFrame((_, delta) => {
    if (mixer) {
      mixer.update(delta);
    }
  });

  if (!scene) {
    return null;
  }

  return (
    <group ref={groupRef}>
      <primitive ref={modelRef} object={scene} />

      {/* Particle Effects - only show when active */}
      {(state.isTalking || state.isThinking) && (
        <Sparkles
          count={50}
          scale={3}
          size={2}
          speed={0.4}
          opacity={0.5}
          color="#ffffff"
        />
      )}
    </group>
  );
};

// Preload models for better performance
useGLTF.preload('/64f1a714fe61576b46f27ca2.glb');
useGLTF.preload('/animations.glb');

export default Avatar;
