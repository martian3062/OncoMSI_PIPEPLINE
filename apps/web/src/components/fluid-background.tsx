"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

type VantaEffectInstance = {
  destroy?: () => void;
};

export function FluidBackground() {
  const hostRef = useRef<HTMLDivElement>(null);
  const effectRef = useRef<VantaEffectInstance | null>(null);

  useEffect(() => {
    let mounted = true;

    async function setup() {
      if (!hostRef.current || effectRef.current) {
        return;
      }

      const [{ default: FOG }] = await Promise.all([import("vanta/dist/vanta.fog.min")]);
      if (!mounted || !hostRef.current) {
        return;
      }

      effectRef.current = FOG({
        el: hostRef.current,
        THREE,
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200,
        minWidth: 200,
        highlightColor: 0xf6fbff,
        midtoneColor: 0xdcebfb,
        lowlightColor: 0xbdd9ef,
        baseColor: 0xf7fbff,
        blurFactor: 0.72,
        speed: 1.45,
        zoom: 0.72,
      }) as VantaEffectInstance;
    }

    setup();

    return () => {
      mounted = false;
      effectRef.current?.destroy?.();
      effectRef.current = null;
    };
  }, []);

  return <div ref={hostRef} className="fluid-scene" aria-hidden="true" />;
}
