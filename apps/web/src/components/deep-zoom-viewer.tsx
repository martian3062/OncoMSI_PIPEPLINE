"use client";

import { Minus, Move, Plus, RotateCcw } from "lucide-react";
import { useEffect, useRef } from "react";
import type OpenSeadragon from "openseadragon";

type DeepZoomViewerProps = {
  src?: string;
  alt: string;
  emptyLabel: string;
  heightClassName?: string;
};

export function DeepZoomViewer({
  src,
  alt,
  emptyLabel,
  heightClassName = "",
}: DeepZoomViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);

  useEffect(() => {
    if (!containerRef.current || !src) {
      if (viewerRef.current) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
      return;
    }

    let isCancelled = false;

    async function loadViewer() {
      const osdModule = await import("openseadragon");
      if (isCancelled || !containerRef.current) {
        return;
      }
      const OpenSeadragon = osdModule.default;
      const viewer = OpenSeadragon({
        element: containerRef.current,
        tileSources: {
          type: "image",
          url: src,
        },
        showNavigator: true,
        showNavigationControl: false,
        visibilityRatio: 1,
        constrainDuringPan: true,
        animationTime: 0.8,
        blendTime: 0.15,
        maxZoomPixelRatio: 12,
        minZoomImageRatio: 0.75,
        navigatorPosition: "BOTTOM_RIGHT",
        navigatorWidth: "26%",
        navigatorHeight: "26%",
      });
      viewerRef.current = viewer;
    }

    void loadViewer();

    return () => {
      isCancelled = true;
      if (viewerRef.current) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  }, [src]);

  function zoomIn() {
    viewerRef.current?.viewport.zoomBy(1.25);
    viewerRef.current?.viewport.applyConstraints();
  }

  function zoomOut() {
    viewerRef.current?.viewport.zoomBy(0.8);
    viewerRef.current?.viewport.applyConstraints();
  }

  function resetView() {
    viewerRef.current?.viewport.goHome(true);
  }

  if (!src) {
    return (
      <div className={`osd-shell ${heightClassName}`}>
        <div className="osd-empty" aria-label={alt}>
          {emptyLabel}
        </div>
      </div>
    );
  }

  return (
    <div className={`osd-shell ${heightClassName}`}>
      <div className="osd-toolbar">
        <button type="button" className="osd-tool" onClick={zoomIn} aria-label="Zoom in">
          <Plus size={16} />
        </button>
        <button type="button" className="osd-tool" onClick={zoomOut} aria-label="Zoom out">
          <Minus size={16} />
        </button>
        <button type="button" className="osd-tool" onClick={resetView} aria-label="Reset view">
          <RotateCcw size={16} />
        </button>
        <span className="osd-hint">
          <Move size={14} />
          Drag to pan
        </span>
      </div>
      <div ref={containerRef} className="osd-stage" aria-label={alt} />
    </div>
  );
}
