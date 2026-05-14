declare module "vanta/dist/vanta.fog.min" {
  type VantaOptions = {
    el: HTMLElement;
    THREE: unknown;
    mouseControls?: boolean;
    touchControls?: boolean;
    gyroControls?: boolean;
    minHeight?: number;
    minWidth?: number;
    highlightColor?: number;
    midtoneColor?: number;
    lowlightColor?: number;
    baseColor?: number;
    blurFactor?: number;
    speed?: number;
    zoom?: number;
  };

  type VantaInstance = {
    destroy?: () => void;
  };

  export default function FOG(options: VantaOptions): VantaInstance;
}
