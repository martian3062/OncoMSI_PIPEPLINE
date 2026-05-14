"use client";

export const dynamic = "force-dynamic";

import { FormEvent, useEffect, useMemo, useRef, useState, startTransition } from "react";
import { AlertCircle, Archive, BarChart3, Clock3, Cpu, HardDrive, LoaderCircle, MemoryStick, ScanSearch, Sparkles, Upload } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { DeepZoomViewer } from "@/components/deep-zoom-viewer";
import { FluidBackground } from "@/components/fluid-background";

type CheckpointTrace = {
  checkpoint: string;
  repeat: number;
  fold: number;
  probability: number;
  threshold: number;
  auroc: number;
  f1_macro: number;
  auprc: number;
  balanced_accuracy: number;
  quality_score: number;
};

type InferenceMetadata = {
  pipeline_mode: string;
  pipeline_style: string;
  device: string;
  preferred_device: string;
  cuda_available: boolean;
  gpu_status: string;
  mean_threshold: number;
  encoder: {
    encoder_label?: string;
    encoder_type?: string;
    embedding_dim?: number;
    tile_count?: number;
    backbone_name?: string;
  };
};

type RuntimeSystem = {
  hostname: string;
  system: string;
  release: string;
  machine: string;
  python_version: string;
  platform_summary: string;
  base_dir: string;
  disk_total_gib: number;
  disk_free_gib: number;
  disk_used_gib: number;
  prefer_device: string;
  pipeline_mode: string;
  max_inference_tiles: number;
  ram: {
    total_gib: number | null;
    available_gib: number | null;
    used_gib: number | null;
  };
  gpu: {
    cuda_available: boolean;
    device_count: number;
    name: string;
    memory_total_gib: number | null;
    memory_reserved_gib: number | null;
    memory_allocated_gib: number | null;
    driver_line: string;
  };
};

type PredictMetadata = {
  inference: InferenceMetadata;
  system: RuntimeSystem;
};

type PredictJobStatus = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  uploaded_name?: string;
  error?: string;
  elapsed_seconds?: number;
  eta_seconds?: number | null;
  progress?: {
    stage: string;
    label: string;
    detail: string;
    percent: number;
  };
  result?: PredictResponse | null;
  inference?: InferenceMetadata;
  system?: RuntimeSystem;
};

type PredictResponse = {
  label: string;
  confidence_level: string;
  confidence_percent: number;
  tile_count: number;
  threshold: number;
  probability: number;
  confidence_score: number;
  model_quality_score: number;
  vote_strength_score: number;
  feature_dim: number;
  checkpoint_count: number;
  input_kind: string;
  input_kind_display?: string;
  encoder_label?: string;
  encoder_backbone?: string;
  encoder_type?: string;
  per_checkpoint: CheckpointTrace[];
  specimen_preview_data_url?: string;
  tile_preview_data_url?: string;
  inference?: InferenceMetadata;
  system?: RuntimeSystem;
};

type AnnotationLabel = "" | "MSS" | "MSI-H";

type StorageSample = {
  patient: string;
  msi_status: string;
  bucket_name: string;
  available_on_vm: boolean;
  source_group: string;
  specimen_preview_data_url?: string;
  tile_preview_data_url?: string;
};

type StorageManifest = {
  title: string;
  summary: {
    requested_total: number;
    available_total: number;
    msi_h_total: number;
    mss_total: number;
    note?: string;
  };
  files: StorageSample[];
};

type StorageSampleDetail = StorageSample & {
  specimen_preview_data_url?: string;
  tile_preview_data_url?: string;
};

type PredictionHistoryItem = {
  job_id?: string;
  saved_at: string;
  uploaded_name: string;
  pipeline_mode: string;
  label: string;
  expected_label?: string;
  patient?: string;
  source_group?: string;
  confidence_level: string;
  confidence_percent: number;
  probability: number;
  threshold: number;
  tile_count: number;
  checkpoint_count: number;
  feature_dim: number;
  input_kind: string;
  input_kind_display?: string;
  encoder_label?: string;
  encoder_backbone?: string;
  encoder_type?: string;
  confidence_score?: number;
  model_quality_score?: number;
  vote_strength_score?: number;
  elapsed_seconds: number;
  specimen_preview_data_url?: string;
  tile_preview_data_url?: string;
  per_checkpoint?: CheckpointTrace[];
  inference?: InferenceMetadata;
  system?: RuntimeSystem;
  result_payload?: PredictResponse & {
    uploaded_name?: string;
  };
};

type PredictionHistoryPayload = {
  count: number;
  items: PredictionHistoryItem[];
};

type AnalysisDatum = {
  name: string;
  total?: number;
  correct?: number;
  wrong?: number;
  accuracy?: number;
  count?: number;
};

type AnalysisPayload = {
  overview: {
    total_scored: number;
    correct: number;
    wrong: number;
    accuracy: number;
    false_positive: number;
    false_negative: number;
    type_i_error: number;
    type_ii_error: number;
  };
  confusion: {
    MSS_to_MSS: number;
    MSS_to_MSI_H: number;
    MSI_H_to_MSI_H: number;
    MSI_H_to_MSS: number;
  };
  by_source_group: AnalysisDatum[];
  by_slide_type: AnalysisDatum[];
  by_pipeline_mode: AnalysisDatum[];
  confidence_distribution: AnalysisDatum[];
  batch_progress: Array<{
    batch_name: string;
    selected_total: number;
    completed_total: number;
    status: string;
    has_autostart: boolean;
  }>;
  current_model: {
    pipeline_mode?: string;
    pipeline_style?: string;
    approach_label?: string;
    mil_model?: string;
    encoder_label?: string;
    feature_dim?: number;
    selected_checkpoint_count?: number;
    available_checkpoints?: number;
    mean_threshold?: number;
  };
  recent_wrong_cases: Array<{
    uploaded_name?: string;
    patient?: string;
    expected_label?: string;
    label?: string;
    probability?: number;
    confidence_level?: string;
    source_group?: string;
    saved_at?: string;
    slide_type?: string;
  }>;
  updated_at: string;
};

type BatchStatusPayload = {
  has_active_batch: boolean;
  batch_name: string;
  phase: string;
  phase_label: string;
  current_index: number;
  total: number;
  completed: number;
  percent: number;
  current_file: string;
  queued_batches: string[];
  updated_at: string;
};

type ProcessingStage = {
  label: string;
  detail: string;
  threshold: number;
};

type LiveStageRecord = {
  stage: string;
  label: string;
  detail: string;
  percent: number;
};

type JobOrigin = "upload" | "storage" | null;

const BROWSER_RENDERABLE_IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "bmp", "webp", "gif"]);

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [annotationLabel, setAnnotationLabel] = useState<AnnotationLabel>("");
  const [activeExpectedLabel, setActiveExpectedLabel] = useState<AnnotationLabel>("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [metadata, setMetadata] = useState<PredictMetadata | null>(null);
  const [storageManifest, setStorageManifest] = useState<StorageManifest | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<PredictionHistoryPayload | null>(null);
  const [analysisSummary, setAnalysisSummary] = useState<AnalysisPayload | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatusPayload | null>(null);
  const [selectedStorageDetail, setSelectedStorageDetail] = useState<StorageSampleDetail | null>(null);
  const [job, setJob] = useState<PredictJobStatus | null>(null);
  const [activeTab, setActiveTab] = useState<"predict" | "storage" | "history" | "analysis">("predict");
  const [predictMode, setPredictMode] = useState<"exact" | "fast">(() => {
    if (typeof window === "undefined") {
      return "exact";
    }
    const mode = new URLSearchParams(window.location.search).get("mode");
    return mode === "fast" ? "fast" : "exact";
  });
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [deletingHistoryKey, setDeletingHistoryKey] = useState("");
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [stageHistory, setStageHistory] = useState<LiveStageRecord[]>([]);
  const [jobOrigin, setJobOrigin] = useState<JobOrigin>(null);
  const [resultSourceName, setResultSourceName] = useState("");
  const [highlightedHistoryKey, setHighlightedHistoryKey] = useState("");
  const [pendingHistoryLookup, setPendingHistoryLookup] = useState("");
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<PredictionHistoryItem | null>(null);
  const [selectedStorageBucket, setSelectedStorageBucket] = useState("");
  const pollFailureCountRef = useRef(0);
  const directPredictApi = useMemo(() => {
    if (typeof window === "undefined") {
      return "/api/predict";
    }
    return `${window.location.protocol}//${window.location.hostname}:8000/api/predict-jobs/`;
  }, []);
  const storageTestApi = "/api/storage-test";

  const selectedName = useMemo(() => file?.name || resultSourceName || "No specimen selected", [file, resultSourceName]);
  const localPreviewUrl = useMemo(() => {
    const ext = file?.name.split(".").pop()?.toLowerCase() || "";
    if (!file || !BROWSER_RENDERABLE_IMAGE_EXTS.has(ext)) {
      return "";
    }
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    let isMounted = true;
    async function loadMetadata() {
      try {
        const response = await fetch(`/api/predict?mode=${predictMode}`, { cache: "no-store" });
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as PredictMetadata;
        if (isMounted) {
          setMetadata(payload);
        }
      } catch {
        // No-op; the page can still function without the metadata surface.
      }
    }
    void loadMetadata();
    return () => {
      isMounted = false;
    };
  }, [predictMode]);

  useEffect(() => {
    let isMounted = true;
    async function loadLibraryData() {
      try {
        const [storageResponse, historyResponse, batchStatusResponse, analysisResponse] = await Promise.all([
          fetch("/api/storage?compact=1", { cache: "no-store" }),
          fetch("/api/history?compact=1", { cache: "no-store" }),
          fetch("/api/batch-status", { cache: "no-store" }),
          fetch("/api/analysis", { cache: "no-store" }),
        ]);
        if (isMounted && storageResponse.ok) {
          setStorageManifest((await storageResponse.json()) as StorageManifest);
        }
        if (isMounted && historyResponse.ok) {
          setPredictionHistory((await historyResponse.json()) as PredictionHistoryPayload);
        }
        if (isMounted && batchStatusResponse.ok) {
          setBatchStatus((await batchStatusResponse.json()) as BatchStatusPayload);
        }
        if (isMounted && analysisResponse.ok) {
          setAnalysisSummary((await analysisResponse.json()) as AnalysisPayload);
        }
      } catch {
        // Keep the tabs graceful if the helper feeds are unavailable.
      }
    }
    void loadLibraryData();
    return () => {
      isMounted = false;
    };
  }, []);

  async function refreshHistory() {
    try {
      const response = await fetch("/api/history?compact=1", { cache: "no-store" });
      if (!response.ok) {
        return;
      }
      setPredictionHistory((await response.json()) as PredictionHistoryPayload);
    } catch {
      // Best effort refresh only.
    }
  }

  async function refreshAnalysis() {
    try {
      const response = await fetch("/api/analysis", { cache: "no-store" });
      if (!response.ok) {
        return;
      }
      setAnalysisSummary((await response.json()) as AnalysisPayload);
    } catch {
      // Best effort refresh only.
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function loadBatchStatus() {
      try {
        const response = await fetch("/api/batch-status", { cache: "no-store" });
        if (!response.ok || cancelled) {
          return;
        }
        setBatchStatus((await response.json()) as BatchStatusPayload);
      } catch {
        // Best effort only; the rest of the page can stay live without it.
      }
    }

    void loadBatchStatus();
    const timer = window.setInterval(() => {
      void loadBatchStatus();
    }, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadAnalysis() {
      try {
        const response = await fetch("/api/analysis", { cache: "no-store" });
        if (!response.ok || cancelled) {
          return;
        }
        setAnalysisSummary((await response.json()) as AnalysisPayload);
      } catch {
        // Best effort only.
      }
    }
    const timer = window.setInterval(() => {
      void loadAnalysis();
    }, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const historyLookup = useMemo(() => {
    const lookup = new Map<string, PredictionHistoryItem>();
    const items = predictionHistory?.items || [];
    for (const item of items) {
      const key = item.uploaded_name.trim().toLowerCase();
      if (!key) {
        continue;
      }
      const existing = lookup.get(key);
      if (!existing || new Date(item.saved_at).getTime() > new Date(existing.saved_at).getTime()) {
        lookup.set(key, item);
      }
    }
    return lookup;
  }, [predictionHistory]);

  const selectedStorageSample = useMemo(() => {
    const files = storageManifest?.files || [];
    if (!files.length) {
      return null;
    }
    const directMatch = files.find((item) => item.bucket_name === selectedStorageBucket);
    return directMatch || files[0];
  }, [selectedStorageBucket, storageManifest]);

  const activeStorageSample = useMemo<StorageSampleDetail | StorageSample | null>(() => {
    if (!selectedStorageSample) {
      return null;
    }
    if (selectedStorageDetail?.bucket_name === selectedStorageSample.bucket_name) {
      return { ...selectedStorageSample, ...selectedStorageDetail };
    }
    return selectedStorageSample;
  }, [selectedStorageDetail, selectedStorageSample]);

  const liveBatchStatusLine = useMemo(() => {
    if (!batchStatus?.has_active_batch) {
      return "No active staged batch";
    }
    const progressLabel = batchStatus.phase === "predicting"
      ? `${batchStatus.completed}/${batchStatus.total} predicted`
      : `${batchStatus.current_index}/${batchStatus.total} downloaded`;
    return `${batchStatus.phase_label} • ${progressLabel}`;
  }, [batchStatus]);

  const sourceAccuracyChartData = useMemo(
    () => (analysisSummary?.by_source_group || []).map((item) => ({ ...item, accuracyPercent: Number(((item.accuracy || 0) * 100).toFixed(1)) })),
    [analysisSummary],
  );

  const slideTypeChartData = useMemo(
    () => (analysisSummary?.by_slide_type || []).map((item) => ({ ...item, accuracyPercent: Number(((item.accuracy || 0) * 100).toFixed(1)) })),
    [analysisSummary],
  );

  function historyKeyFor(item: PredictionHistoryItem) {
    return item.job_id || `${item.saved_at}-${item.uploaded_name}`;
  }

  function historyAnchorFor(item: PredictionHistoryItem) {
    return `history-${encodeURIComponent(historyKeyFor(item))}`;
  }

  function expectedComparisonLabel(expectedLabel: string, predictedLabel: string) {
    if (!expectedLabel) {
      return "";
    }
    return expectedLabel === predictedLabel ? "Matched expected" : "Different from expected";
  }

  function slideTypeForName(name: string) {
    const upper = (name || "").trim().toUpperCase();
    if (!upper) {
      return "";
    }
    const stem = upper.endsWith(".SVS") ? upper.slice(0, -4) : upper;
    const parts = stem.split("-");
    for (const part of parts) {
      if (part.startsWith("DX") || part.startsWith("TS") || part.startsWith("BS") || part.startsWith("MS")) {
        return part.split(".")[0];
      }
    }
    return "";
  }

  function focusHistoryResult(item: PredictionHistoryItem) {
    const historyKey = historyKeyFor(item);
    const anchorId = historyAnchorFor(item);
    setActiveTab("history");
    setHighlightedHistoryKey(historyKey);
    setSelectedHistoryItem(item);
    window.setTimeout(() => {
      document.getElementById(anchorId)?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }, 80);
  }

  async function fetchHistoryDetail(item: PredictionHistoryItem) {
    const query = item.job_id
      ? `job_id=${encodeURIComponent(item.job_id)}`
      : `saved_at=${encodeURIComponent(item.saved_at)}&uploaded_name=${encodeURIComponent(item.uploaded_name)}`;
    const response = await fetch(`/api/history?${query}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Could not load the saved result details.");
    }
    const detailedItem = (await response.json()) as PredictionHistoryItem;
    setSelectedHistoryItem(detailedItem);
    return detailedItem;
  }

  function focusHistoryByUploadedName(uploadedName: string) {
    const match = historyLookup.get(uploadedName.trim().toLowerCase());
    if (!match) {
      return;
    }
    focusHistoryResult(match);
  }

  function openHistoryResult(item: PredictionHistoryItem) {
    const restore = (source: PredictionHistoryItem) => {
      const restoredResult: PredictResponse = source.result_payload || {
        label: source.label,
        confidence_level: source.confidence_level,
        confidence_percent: Number(source.confidence_percent || 0),
        tile_count: Number(source.tile_count || 0),
        threshold: Number(source.threshold || 0),
        probability: Number(source.probability || 0),
        confidence_score: Number(source.confidence_score || 0),
        model_quality_score: Number(source.model_quality_score || 0),
        vote_strength_score: Number(source.vote_strength_score || 0),
        feature_dim: Number(source.feature_dim || 0),
        checkpoint_count: Number(source.checkpoint_count || 0),
        input_kind: source.input_kind || "",
        input_kind_display: source.input_kind_display || "",
        encoder_label: source.encoder_label || "",
        encoder_backbone: source.encoder_backbone || "",
        encoder_type: source.encoder_type || "",
        per_checkpoint: source.per_checkpoint || [],
        specimen_preview_data_url: source.specimen_preview_data_url || "",
        tile_preview_data_url: source.tile_preview_data_url || "",
        inference: source.inference,
        system: source.system,
      };
      setFile(null);
      setActiveExpectedLabel((source.expected_label as AnnotationLabel) || "");
      setResult(restoredResult);
      setResultSourceName(source.patient || source.uploaded_name);
      if (source.inference && source.system) {
        setMetadata({ inference: source.inference, system: source.system });
      }
      setError("");
      setJob(null);
      setIsLoading(false);
      setStartedAt(null);
      setElapsedSeconds(0);
      setStageHistory([]);
      setActiveTab("predict");
      setHighlightedHistoryKey(historyKeyFor(source));
      setSelectedHistoryItem(source);
      if (typeof window !== "undefined") {
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    };
    if (!item.per_checkpoint?.length && !item.specimen_preview_data_url && !item.tile_preview_data_url) {
      void fetchHistoryDetail(item)
        .then((detailedItem) => {
          restore(detailedItem);
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : "Could not load the saved result details.");
        });
      return;
    }
    const restoredResult: PredictResponse = item.result_payload || {
      label: item.label,
      confidence_level: item.confidence_level,
      confidence_percent: Number(item.confidence_percent || 0),
      tile_count: Number(item.tile_count || 0),
      threshold: Number(item.threshold || 0),
      probability: Number(item.probability || 0),
      confidence_score: Number(item.confidence_score || 0),
      model_quality_score: Number(item.model_quality_score || 0),
      vote_strength_score: Number(item.vote_strength_score || 0),
      feature_dim: Number(item.feature_dim || 0),
      checkpoint_count: Number(item.checkpoint_count || 0),
      input_kind: item.input_kind || "",
      input_kind_display: item.input_kind_display || "",
      encoder_label: item.encoder_label || "",
      encoder_backbone: item.encoder_backbone || "",
      encoder_type: item.encoder_type || "",
      per_checkpoint: item.per_checkpoint || [],
      specimen_preview_data_url: item.specimen_preview_data_url || "",
      tile_preview_data_url: item.tile_preview_data_url || "",
      inference: item.inference,
      system: item.system,
    };
    void restoredResult;
    restore(item);
  }

  async function handleDeleteHistory(item: PredictionHistoryItem) {
    const historyKey = historyKeyFor(item);
    setDeletingHistoryKey(historyKey);
    try {
      const response = await fetch("/api/history", {
        method: "DELETE",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({
          job_id: item.job_id || "",
          saved_at: item.saved_at,
          uploaded_name: item.uploaded_name,
        }),
      });
      if (!response.ok) {
        throw new Error("Could not delete the saved record.");
      }
      await refreshHistory();
      await refreshAnalysis();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete the saved record.");
    } finally {
      setDeletingHistoryKey("");
    }
  }

  useEffect(() => {
    if (!pendingHistoryLookup || activeTab !== "history") {
      return;
    }
    const matchedItem = historyLookup.get(pendingHistoryLookup.trim().toLowerCase());
    if (!matchedItem) {
      return;
    }
    focusHistoryResult(matchedItem);
    setPendingHistoryLookup("");
  }, [activeTab, historyLookup, pendingHistoryLookup]);

  useEffect(() => {
    if (!selectedHistoryItem?.uploaded_name) {
      return;
    }
    if (selectedHistoryItem.per_checkpoint?.length || selectedHistoryItem.specimen_preview_data_url || selectedHistoryItem.tile_preview_data_url) {
      return;
    }
    void fetchHistoryDetail(selectedHistoryItem).catch(() => {
      // Keep the compact item selected if the detail fetch fails.
    });
  }, [selectedHistoryItem]);

  useEffect(() => {
    const files = storageManifest?.files || [];
    if (!files.length) {
      setSelectedStorageBucket("");
      return;
    }
    if (!selectedStorageBucket || !files.some((item) => item.bucket_name === selectedStorageBucket)) {
      setSelectedStorageBucket(files[0].bucket_name);
    }
  }, [selectedStorageBucket, storageManifest]);

  useEffect(() => {
    const bucketName = selectedStorageSample?.bucket_name || "";
    if (!bucketName) {
      setSelectedStorageDetail(null);
      return;
    }
    let cancelled = false;
    async function loadStorageDetail() {
      try {
        const response = await fetch(`/api/storage?bucket_name=${encodeURIComponent(bucketName)}`, { cache: "no-store" });
        if (!response.ok || cancelled) {
          return;
        }
        setSelectedStorageDetail((await response.json()) as StorageSampleDetail);
      } catch {
        // Keep storage usable even if the preview detail fetch fails.
      }
    }
    void loadStorageDetail();
    return () => {
      cancelled = true;
    };
  }, [selectedStorageSample?.bucket_name]);

  useEffect(() => {
    if (!predictionHistory?.items?.length) {
      setSelectedHistoryItem(null);
      return;
    }
    if (!selectedHistoryItem) {
      return;
    }
    const selectedKey = historyKeyFor(selectedHistoryItem);
    const refreshedMatch = predictionHistory.items.find((item) => historyKeyFor(item) === selectedKey);
    setSelectedHistoryItem(refreshedMatch || null);
  }, [predictionHistory, selectedHistoryItem]);

  useEffect(() => {
    if (!isLoading || startedAt === null) {
      return;
    }
    const timer = window.setInterval(() => {
      const liveElapsed = Math.floor((Date.now() - startedAt) / 1000);
      setElapsedSeconds((previous) => Math.max(previous, liveElapsed));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [isLoading, startedAt]);

  useEffect(() => {
    return () => {
      if (localPreviewUrl) {
        URL.revokeObjectURL(localPreviewUrl);
      }
    };
  }, [localPreviewUrl]);

  useEffect(() => {
    if (!job?.job_id || !isLoading) {
      return;
    }
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/predict?job_id=${encodeURIComponent(job.job_id)}&mode=${predictMode}`, {
          cache: "no-store",
        });
        const payload = (await response.json()) as PredictJobStatus & { error?: string };
        if (!response.ok) {
          throw new Error(payload.error || "Prediction status check failed.");
        }
        if (cancelled) {
          return;
        }
        pollFailureCountRef.current = 0;
        setJob(payload);
        appendStageRecord(payload);
        if (typeof payload.elapsed_seconds === "number") {
          setElapsedSeconds((previous) => Math.max(previous, payload.elapsed_seconds ?? 0));
        }
        if (payload.status === "completed" && payload.result) {
          setResult(payload.result);
          setResultSourceName(payload.uploaded_name || job?.uploaded_name || "");
          if (payload.inference && payload.system) {
            setMetadata({ inference: payload.inference, system: payload.system });
          }
          void refreshHistory();
          void refreshAnalysis();
          if (jobOrigin === "storage") {
            setPendingHistoryLookup(payload.uploaded_name || job?.uploaded_name || "");
            setActiveTab("history");
          }
          setIsLoading(false);
          setStartedAt(null);
          setJobOrigin(null);
          window.clearInterval(timer);
        } else if (payload.status === "failed") {
          setResult(null);
          setError(payload.error || payload.progress?.detail || "Prediction failed.");
          setIsLoading(false);
          setStartedAt(null);
          setJobOrigin(null);
          window.clearInterval(timer);
        }
      } catch (err) {
        if (cancelled) {
          return;
        }
        pollFailureCountRef.current += 1;
        if (pollFailureCountRef.current < 5) {
          return;
        }
        const message =
          err instanceof Error && err.message
            ? err.message
            : "Prediction status check failed.";
        setError(
          message === "Prediction status check failed."
            ? "The prediction job status could not be refreshed. The backend may have restarted during polling."
            : message,
        );
        setIsLoading(false);
        setStartedAt(null);
        setJobOrigin(null);
        window.clearInterval(timer);
      }
    }, 1500);
    void (async () => {
      try {
        const response = await fetch(`/api/predict?job_id=${encodeURIComponent(job.job_id)}&mode=${predictMode}`, {
          cache: "no-store",
        });
        const payload = (await response.json()) as PredictJobStatus & { error?: string };
        if (response.ok && !cancelled) {
          setJob(payload);
          appendStageRecord(payload);
          if (typeof payload.elapsed_seconds === "number") {
            setElapsedSeconds((previous) => Math.max(previous, payload.elapsed_seconds ?? 0));
          }
        }
      } catch {
        // interval will retry
      }
    })();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [isLoading, job?.job_id, jobOrigin, predictMode]);

  const activeMetadata = result?.inference && result?.system ? { inference: result.inference, system: result.system } : metadata;
  const specimenViewerSrc = result ? result.specimen_preview_data_url || "" : localPreviewUrl;
  const activeSlideType = slideTypeForName(file?.name || resultSourceName || selectedName);
  const probabilityPercent = result ? `${(result.probability * 100).toFixed(2)}%` : "";
  const thresholdPercent = result ? `${(result.threshold * 100).toFixed(2)}%` : "";
  const voteStrengthPercent = result ? `${(result.vote_strength_score * 100).toFixed(1)}%` : "";
  const modelQualityPercent = result ? `${(result.model_quality_score * 100).toFixed(1)}%` : "";
  const confidencePercent = result ? `${result.confidence_percent.toFixed(1)}%` : "";
  const resultComparison = result ? expectedComparisonLabel(activeExpectedLabel, result.label) : "";

  const technicalNarrative = useMemo(() => {
    if (!result) {
      return [];
    }
    const probabilityValue = result.probability * 100;
    const thresholdValue = result.threshold * 100;
    const delta = Math.abs(probabilityValue - thresholdValue);
    const caution =
      result.tile_count < 64
        ? "This was a light local slide pass, so it is faster but less training-matched than the original VM extraction path."
        : "This used a denser tile set, so the local inference pass is closer to the training-side sampling budget.";
    const expectationLine = activeExpectedLabel
      ? `The annotated expected label for this specimen was ${activeExpectedLabel}, and the model ${activeExpectedLabel === result.label ? "matched" : "did not match"} that expectation.`
      : "No expected MSI annotation was attached to this run, so the result is shown without a match check.";
    return [
      `${selectedName} was classified as ${result.label} because the ensemble mean MSI-H probability was ${probabilityValue.toFixed(2)}%, above the operating threshold of ${thresholdValue.toFixed(2)}%.`,
      expectationLine,
      `The confidence bucket is ${result.confidence_level}, with a blended confidence score of ${confidencePercent}. Vote strength is ${voteStrengthPercent}, which reflects how far the averaged probability sits from the decision threshold.`,
      `The model stack used ${result.checkpoint_count} preserved checkpoints with feature dimension ${result.feature_dim} and processed ${result.tile_count} tiles in this inference pass.`,
      `${caution} The current output is slide-level, not a count of positive versus negative tiles.`,
      `Threshold margin on this case was ${delta.toFixed(2)} percentage points, and the model-quality blend behind the selected ensemble was ${modelQualityPercent}.`,
    ];
  }, [activeExpectedLabel, confidencePercent, modelQualityPercent, result, selectedName, voteStrengthPercent]);

  const activeHistoryDetail = useMemo(() => {
    if (!selectedHistoryItem) {
      return null;
    }
    const restoredResult: PredictResponse = selectedHistoryItem.result_payload || {
      label: selectedHistoryItem.label,
      confidence_level: selectedHistoryItem.confidence_level,
      confidence_percent: Number(selectedHistoryItem.confidence_percent || 0),
      tile_count: Number(selectedHistoryItem.tile_count || 0),
      threshold: Number(selectedHistoryItem.threshold || 0),
      probability: Number(selectedHistoryItem.probability || 0),
      confidence_score: Number(selectedHistoryItem.confidence_score || 0),
      model_quality_score: Number(selectedHistoryItem.model_quality_score || 0),
      vote_strength_score: Number(selectedHistoryItem.vote_strength_score || 0),
      feature_dim: Number(selectedHistoryItem.feature_dim || 0),
      checkpoint_count: Number(selectedHistoryItem.checkpoint_count || 0),
      input_kind: selectedHistoryItem.input_kind || "",
      input_kind_display: selectedHistoryItem.input_kind_display || "",
      encoder_label: selectedHistoryItem.encoder_label || "",
      encoder_backbone: selectedHistoryItem.encoder_backbone || "",
      encoder_type: selectedHistoryItem.encoder_type || "",
      per_checkpoint: selectedHistoryItem.per_checkpoint || [],
      specimen_preview_data_url: selectedHistoryItem.specimen_preview_data_url || "",
      tile_preview_data_url: selectedHistoryItem.tile_preview_data_url || "",
      inference: selectedHistoryItem.inference,
      system: selectedHistoryItem.system,
    };
    const historyName = selectedHistoryItem.patient || selectedHistoryItem.uploaded_name;
    const expectedLabel = (selectedHistoryItem.expected_label || "") as AnnotationLabel;
    const probabilityValue = restoredResult.probability * 100;
    const thresholdValue = restoredResult.threshold * 100;
    const delta = Math.abs(probabilityValue - thresholdValue);
    const confidenceValue = `${restoredResult.confidence_percent.toFixed(1)}%`;
    const voteValue = `${(restoredResult.vote_strength_score * 100).toFixed(1)}%`;
    const qualityValue = `${(restoredResult.model_quality_score * 100).toFixed(1)}%`;
    const historyNarrative = [
      `${historyName} was saved as ${restoredResult.label} with an MSI-H probability of ${probabilityValue.toFixed(2)}% against a threshold of ${thresholdValue.toFixed(2)}%.`,
      expectedLabel
        ? `The saved expected label was ${expectedLabel}, and this result ${expectedLabel === restoredResult.label ? "matched" : "did not match"} that annotation.`
        : "No expected label was saved with this history entry.",
      `Saved confidence is ${restoredResult.confidence_level} at ${confidenceValue}. Vote strength is ${voteValue} and the blended model-quality score is ${qualityValue}.`,
      `This rerun kept ${restoredResult.checkpoint_count} preserved checkpoints, feature dimension ${restoredResult.feature_dim}, and ${restoredResult.tile_count} sampled tiles.`,
      `Threshold margin on this saved result was ${delta.toFixed(2)} percentage points.`,
    ];
    return {
      item: selectedHistoryItem,
      result: restoredResult,
      title: historyName,
      confidencePercent: confidenceValue,
      probabilityPercent: `${probabilityValue.toFixed(2)}%`,
      thresholdPercent: `${thresholdValue.toFixed(2)}%`,
      voteStrengthPercent: voteValue,
      modelQualityPercent: qualityValue,
      expectedLabel,
      expectedComparison: expectedComparisonLabel(expectedLabel, restoredResult.label),
      technicalNarrative: historyNarrative,
    };
  }, [selectedHistoryItem]);

  const processingStages = useMemo<ProcessingStage[]>(() => {
    const ext = file?.name.split(".").pop()?.toLowerCase() || "";
    const isSlide = ["svs", "tif", "tiff", "ndpi", "mrxs", "scn"].includes(ext);
    const isExact = (activeMetadata?.inference.pipeline_mode || "").toLowerCase() === "manager1";
    return [
      { label: "Upload received", detail: "The specimen file has been accepted by the predictor.", threshold: 0 },
      {
        label: "Preview decode",
        detail: "Generating the visible specimen preview and preparing the request payload.",
        threshold: 5,
      },
      {
        label: isSlide && isExact ? "Slideflow tile extraction" : isSlide ? "Slide tissue reading" : "Image tile sampling",
        detail: isSlide && isExact
          ? "Reading the WSI and cutting the training-matched tile set through the exact slide pipeline."
          : isSlide
            ? "Reading the slide and selecting tissue-rich regions for tile extraction."
            : "Sampling tiles from the uploaded image for encoder input.",
        threshold: isExact ? 35 : 14,
      },
      {
        label: isSlide && isExact ? "Feature bag generation" : "Virchow2 encoding",
        detail: isSlide && isExact
          ? "Writing the temporary feature bag that will be scored by the preserved ensemble."
          : "Encoding the selected tiles into Virchow2 feature vectors.",
        threshold: isExact ? 110 : 24,
      },
      {
        label: "TransMIL ensemble scoring",
        detail: "Running the preserved fold checkpoints and averaging the MSI-H probability.",
        threshold: isExact ? 165 : 34,
      },
      {
        label: "Result packaging",
        detail: "Preparing the final response, previews, and numeric technical record.",
        threshold: isExact ? 190 : 42,
      },
    ];
  }, [activeMetadata?.inference.pipeline_mode, file?.name]);

  const currentStageIndex = useMemo(() => {
    if (!isLoading) {
      return -1;
    }
    const progressLabel = job?.progress?.label;
    if (progressLabel) {
      const matchedIndex = processingStages.findIndex((item) => item.label === progressLabel);
      if (matchedIndex >= 0) {
        return matchedIndex;
      }
    }
    let idx = 0;
    for (let i = 0; i < processingStages.length; i += 1) {
      if (elapsedSeconds >= processingStages[i].threshold) {
        idx = i;
      }
    }
    return idx;
  }, [elapsedSeconds, isLoading, job?.progress?.label, processingStages]);

  const progressPercent = (() => {
    if (!isLoading) {
      return result ? 100 : 0;
    }
    if (typeof job?.progress?.percent === "number") {
      return Math.max(0, Math.min(100, job.progress.percent));
    }
    return Math.min(96, Math.max(6, Math.round((elapsedSeconds / Math.max(processingStages.at(-1)?.threshold || 1, 1)) * 100)));
  })();

  const liveStageItems = useMemo(() => {
    if (stageHistory.length > 0) {
      return stageHistory;
    }
    if (job?.progress?.label) {
      return [{
        stage: job.progress?.stage || "running",
        label: job.progress?.label || "Running",
        detail: job.progress?.detail || "The current prediction step is running.",
        percent: typeof job.progress?.percent === "number" ? job.progress.percent : progressPercent,
      }];
    }
    return [{
      stage: `estimated-${currentStageIndex}`,
      label: processingStages[Math.max(currentStageIndex, 0)]?.label || "Preparing request",
      detail: processingStages[Math.max(currentStageIndex, 0)]?.detail || "Preparing the current inference request.",
      percent: progressPercent,
    }];
  }, [currentStageIndex, job?.progress?.detail, job?.progress?.label, job?.progress?.percent, job?.progress?.stage, processingStages, progressPercent, stageHistory]);

  const currentLiveStage = job?.progress?.label || liveStageItems.at(-1)?.label || "Preparing request";
  const activeDisplayStageIndex = liveStageItems.length - 1;

  function formatEta(totalSeconds: number) {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    if (minutes <= 0) {
      return `${seconds}s`;
    }
    return `${minutes}m ${seconds}s`;
  }

  function switchPredictMode(nextMode: "exact" | "fast") {
    if (nextMode === predictMode) {
      return;
    }
    setIsLoading(false);
    setStartedAt(null);
    setElapsedSeconds(0);
    setError("");
    setFile(null);
    setResult(null);
    setJob(null);
    setStageHistory([]);
    setMetadata(null);
    startTransition(() => {
      setPredictMode(nextMode);
    });
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("mode", nextMode);
      window.location.assign(url.toString());
    }
  }

  function formatNumber(value?: number | null, unit = "") {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return "-";
    }
    return `${value}${unit}`;
  }

  function appendStageRecord(payload?: PredictJobStatus | null) {
    if (!payload?.progress?.stage) {
      return;
    }
    setStageHistory((previous) => {
      const nextRecord: LiveStageRecord = {
        stage: payload.progress?.stage || "running",
        label: payload.progress?.label || "Running",
        detail: payload.progress?.detail || "The current prediction step is running.",
        percent: typeof payload.progress?.percent === "number" ? payload.progress.percent : 0,
      };
      const existingIndex = previous.findIndex((item) => item.stage === nextRecord.stage);
      if (existingIndex >= 0) {
        const updated = [...previous];
        updated[existingIndex] = nextRecord;
        return updated;
      }
      return [...previous, nextRecord];
    });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError("Choose a slide or image first.");
      return;
    }
    setIsLoading(true);
    setStartedAt(Date.now());
    setElapsedSeconds(0);
    setError("");
    setResult(null);
    setResultSourceName(file.name);
    setActiveExpectedLabel(annotationLabel);
    setJob(null);
    setStageHistory([]);
    setJobOrigin("upload");
    pollFailureCountRef.current = 0;
    const formData = new FormData();
    formData.append("prediction_input", file);
    if (annotationLabel) {
      formData.append("expected_label", annotationLabel);
    }
    try {
      const response = await fetch(`${directPredictApi}?mode=${predictMode}`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as PredictJobStatus & { error?: string };
      if (!response.ok) {
        throw new Error(payload.error || "Prediction failed.");
      }
      setJob(payload);
      appendStageRecord(payload);
      if (payload.status === "failed") {
        setError(payload.error || payload.progress?.detail || "Prediction failed.");
        setIsLoading(false);
        setStartedAt(null);
        setJobOrigin(null);
      } else if (payload.status === "completed" && payload.result) {
        setResult(payload.result);
        setResultSourceName(payload.uploaded_name || file.name);
        if (payload.inference && payload.system) {
          setMetadata({ inference: payload.inference, system: payload.system });
        }
        void refreshHistory();
        void refreshAnalysis();
        setIsLoading(false);
        setStartedAt(null);
        setJobOrigin(null);
      }
    } catch (err) {
      setResult(null);
      setJob(null);
      setError(err instanceof Error ? err.message : "Prediction failed.");
      setIsLoading(false);
      setStartedAt(null);
      setJobOrigin(null);
    }
  }

  async function handleStorageTest(sample: StorageSample) {
    setActiveTab("predict");
    setIsLoading(true);
    setStartedAt(Date.now());
    setElapsedSeconds(0);
    setError("");
    setResult(null);
    setJob(null);
    setStageHistory([]);
    setFile(null);
    setActiveExpectedLabel((sample.msi_status as AnnotationLabel) || "");
    setResultSourceName(sample.patient || sample.bucket_name);
    setJobOrigin("storage");
    setPendingHistoryLookup(sample.bucket_name);
    pollFailureCountRef.current = 0;
    try {
      const formData = new FormData();
      formData.append("bucket_name", sample.bucket_name);
      const response = await fetch(`${storageTestApi}?mode=${predictMode}`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as PredictJobStatus & { error?: string };
      if (!response.ok) {
        throw new Error(payload.error || "Could not queue stored sample.");
      }
      setJob(payload);
      appendStageRecord(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not queue stored sample.");
      setIsLoading(false);
      setStartedAt(null);
      setJobOrigin(null);
    }
  }

  function formatSavedAt(value: string) {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return value;
    }
    return parsed.toLocaleString();
  }

  function formatAccuracy(value: number | undefined) {
    return `${((value || 0) * 100).toFixed(1)}%`;
  }

  return (
    <main className="app-shell">
      <FluidBackground />
      <div className="app-content">
        <section className="hero-card glass-card">
          <video
            className="hero-card-video"
            src="/hero-background.mp4"
            autoPlay
            muted
            loop
            playsInline
            onLoadedMetadata={(event) => {
              event.currentTarget.playbackRate = 0.85;
            }}
          />
          <div className="hero-card-video-scrim" />
          <div className="hero-layout">
            <div className="hero-copy">
              <span className="hero-pill">
                <Sparkles size={14} />
                Virchow2 live predictor
              </span>
              <h1>MSI detection</h1>
              <p>Fluid MSI detection with real local inference.</p>
            </div>
            <aside className="hero-status-card">
              <div className="hero-status-top">
                <span className={`state-pill ${batchStatus?.has_active_batch ? "state-pill-live" : ""}`}>
                  {batchStatus?.has_active_batch ? "Batch live" : "Idle"}
                </span>
                <strong>{batchStatus?.batch_name || "No staged batch"}</strong>
              </div>
              <p>{liveBatchStatusLine}</p>
              <div className="hero-status-progress">
                <span style={{ width: `${batchStatus?.has_active_batch ? Math.max(6, batchStatus?.percent || 0) : 0}%` }} />
              </div>
              <div className="hero-status-meta">
                <span>{batchStatus?.percent ?? 0}%</span>
                <span>{batchStatus?.queued_batches?.length ? `${batchStatus.queued_batches.length} queued` : "No queue"}</span>
              </div>
              {batchStatus?.current_file ? (
                <div className="hero-status-file">{batchStatus.current_file}</div>
              ) : null}
            </aside>
          </div>
        </section>
        <section className="surface-tabs glass-card">
          <button
            type="button"
            className={`surface-tab ${activeTab === "predict" ? "active" : ""}`}
            onClick={() => setActiveTab("predict")}
          >
            <ScanSearch size={16} />
            Predict
          </button>
          <button
            type="button"
            className={`surface-tab ${activeTab === "storage" ? "active" : ""}`}
            onClick={() => setActiveTab("storage")}
          >
            <Archive size={16} />
            Storage
          </button>
          <button
            type="button"
            className={`surface-tab ${activeTab === "history" ? "active" : ""}`}
            onClick={() => setActiveTab("history")}
          >
            <Clock3 size={16} />
            History
          </button>
          <button
            type="button"
            className={`surface-tab ${activeTab === "analysis" ? "active" : ""}`}
            onClick={() => setActiveTab("analysis")}
          >
            <BarChart3 size={16} />
            Analysis
          </button>
        </section>

        {activeTab === "predict" ? (
        <section className="workspace-grid">
          <form className="upload-card glass-card" onSubmit={handleSubmit}>
            <div className="section-top">
              <div>
                <p className="section-kicker">Upload</p>
                <h2>Drop one specimen</h2>
              </div>
              <Upload size={20} />
            </div>

            <div className="mode-toggle" role="tablist" aria-label="Prediction mode">
              <button
                type="button"
                className={`mode-toggle-button ${predictMode === "exact" ? "active" : ""}`}
                onClick={() => switchPredictMode("exact")}
                aria-pressed={predictMode === "exact"}
              >
                <strong>Exact mode</strong>
                <span>Accuracy-first exact pipeline</span>
              </button>
              <button
                type="button"
                className={`mode-toggle-button ${predictMode === "fast" ? "active" : ""}`}
                onClick={() => switchPredictMode("fast")}
                aria-pressed={predictMode === "fast"}
              >
                <strong>Fast mode</strong>
                <span>Lower-latency local path</span>
              </button>
            </div>

            <p className="mode-note">
              Current selection: <strong>{predictMode === "exact" ? "Exact mode" : "Fast mode"}</strong>.
              {" "}The runtime details below refresh to match this choice.
            </p>

            <label className="upload-zone">
              <input
                type="file"
                name="prediction_input"
                accept=".svs,.tif,.tiff,.ndpi,.mrxs,.scn,.png,.jpg,.jpeg,.bmp,.webp,.pt,.pth,.bin,.npy,.npz"
                onChange={(event) => {
                  setFile(event.target.files?.[0] || null);
                  setError("");
                }}
              />
              <span className="upload-zone-kicker">Selected file</span>
              <strong>{selectedName}</strong>
              <span className="upload-zone-copy">
                Supports SVS slides, images, and trusted feature bags.
              </span>
            </label>

            <label className="annotation-field">
              <span className="annotation-label">Expected annotation</span>
              <select
                name="expected_label"
                value={annotationLabel}
                onChange={(event) => setAnnotationLabel(event.target.value as AnnotationLabel)}
              >
                <option value="">No annotation</option>
                <option value="MSS">MSS</option>
                <option value="MSI-H">MSI-H</option>
              </select>
              <span className="annotation-help">Save the known MSI status so History can show expected versus predicted for this VM run.</span>
            </label>

            <button className="primary-button" type="submit" disabled={isLoading}>
              {isLoading ? (
                <>
                  <LoaderCircle className="spin" size={18} />
                  Running prediction
                </>
              ) : (
                <>
                  <ScanSearch size={18} />
                  Start prediction
                </>
              )}
            </button>

            {error ? (
              <div className="error-banner">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            ) : null}

            <div className="runtime-panel">
              <div className="runtime-panel-header">
                <div>
                  <p className="section-kicker">Runtime</p>
                  <h3>System and device</h3>
                </div>
                <span className="runtime-badge">
                  {activeMetadata?.inference.cuda_available ? "CUDA ready" : "CPU mode"}
                </span>
              </div>

              {activeMetadata ? (
                <>
                  <div className="runtime-grid">
                    <div className="runtime-box">
                      <span>Architecture</span>
                      <strong>{activeMetadata.system.machine}</strong>
                    </div>
                    <div className="runtime-box">
                      <span>System</span>
                      <strong>{activeMetadata.system.system} {activeMetadata.system.release}</strong>
                    </div>
                    <div className="runtime-box">
                      <span>Resolved device</span>
                      <strong>{activeMetadata.inference.device}</strong>
                    </div>
                    <div className="runtime-box">
                      <span>Preferred device</span>
                      <strong>{activeMetadata.inference.preferred_device}</strong>
                    </div>
                    <div className="runtime-box">
                      <span>GPU</span>
                      <strong>{activeMetadata.system.gpu.name || activeMetadata.inference.gpu_status}</strong>
                    </div>
                    <div className="runtime-box">
                      <span>CUDA status</span>
                      <strong>{activeMetadata.inference.cuda_available ? "Available" : "Not available"}</strong>
                    </div>
                    <div className="runtime-box">
                      <span><MemoryStick size={14} /> RAM free</span>
                      <strong>{formatNumber(activeMetadata.system.ram.available_gib, " GiB")}</strong>
                    </div>
                    <div className="runtime-box">
                      <span><HardDrive size={14} /> Disk free</span>
                      <strong>{formatNumber(activeMetadata.system.disk_free_gib, " GiB")}</strong>
                    </div>
                    <div className="runtime-box">
                      <span><Cpu size={14} /> Pipeline mode</span>
                      <strong>{activeMetadata.inference.pipeline_mode}</strong>
                    </div>
                    <div className="runtime-box">
                      <span>Pipeline style</span>
                      <strong>{activeMetadata.inference.pipeline_style}</strong>
                    </div>
                    <div className="runtime-box">
                      <span>Encoder</span>
                      <strong>{activeMetadata.inference.encoder.encoder_label || "Virchow2"}</strong>
                    </div>
                    <div className="runtime-box">
                      <span>Max tiles</span>
                      <strong>{activeMetadata.system.max_inference_tiles}</strong>
                    </div>
                  </div>
                  <p className="runtime-footnote">
                    Host {activeMetadata.system.hostname} using Python {activeMetadata.system.python_version}. Base path:
                    {" "}{activeMetadata.system.base_dir}
                  </p>
                </>
              ) : (
                <p className="runtime-footnote">Runtime details will load here when the backend responds.</p>
              )}
            </div>
          </form>

          <section className="result-card glass-card">
            <video
              className="result-card-video"
              src="/output-background.mp4"
              autoPlay
              muted
              loop
              playsInline
              onLoadedMetadata={(event) => {
                event.currentTarget.playbackRate = 0.75;
              }}
            />
            <div className="result-card-video-scrim" />
            <div className="section-top">
              <div>
                <p className="section-kicker">Output</p>
                <h2>{result ? "Prediction ready" : "Waiting for result"}</h2>
              </div>
              <span className={`state-pill ${isLoading ? "state-pill-live" : ""}`}>
                {isLoading ? "Processing" : result ? "Complete" : "Idle"}
              </span>
            </div>

            {result ? (
              <>
                <div className="viewer-stack viewer-stack-wide">
                  <div className="viewer-card">
                    <span>Input</span>
                    <DeepZoomViewer
                      src={specimenViewerSrc}
                      alt="Input specimen viewer"
                      emptyLabel="Input preview will appear here."
                      heightClassName="viewer-tall"
                    />
                  </div>

                  <div className="viewer-card">
                    <span>Output tile map</span>
                    <DeepZoomViewer
                      src={result.tile_preview_data_url}
                      alt="Tile sampling viewer"
                      emptyLabel="Tile map will appear here."
                      heightClassName="viewer-grid-height"
                    />
                    <p className="viewer-note">
                      Blue and white markers show the sampled tile regions the model encoded. The alternating colors only separate neighboring picks visually; they do not mean MSI-H versus MSS by themselves.
                    </p>
                  </div>
                </div>

                <div className="result-hero">
                  <div>
                    <span className="result-label">Prediction</span>
                    <strong>{result.label}</strong>
                  </div>
                  <div className="confidence-chip">
                    <span>Confidence</span>
                    <strong>{result.confidence_level}</strong>
                  </div>
                </div>

                {activeSlideType ? (
                  <div className="slide-type-row">
                    <button type="button" className="slide-type-pill" disabled>
                      Slide type: {activeSlideType}
                    </button>
                  </div>
                ) : null}

                {activeExpectedLabel ? (
                  <div className="annotation-summary">
                    <div className="annotation-summary-box">
                      <span>Expected</span>
                      <strong>{activeExpectedLabel}</strong>
                    </div>
                    <div className={`annotation-summary-box ${resultComparison === "Matched expected" ? "match" : "mismatch"}`}>
                      <span>Comparison</span>
                      <strong>{resultComparison}</strong>
                    </div>
                  </div>
                ) : null}

                <div className="result-metrics">
                  <div className="metric-box">
                    <span>Confidence</span>
                    <strong>{confidencePercent}</strong>
                  </div>
                  <div className="metric-box">
                    <span>Tiles used</span>
                    <strong>{result.tile_count}</strong>
                  </div>
                  <div className="metric-box">
                    <span>MSI-H probability</span>
                    <strong>{probabilityPercent}</strong>
                  </div>
                  <div className="metric-box">
                    <span>Decision threshold</span>
                    <strong>{thresholdPercent}</strong>
                  </div>
                  <div className="metric-box">
                    <span>Ensemble checkpoints</span>
                    <strong>{result.checkpoint_count}</strong>
                  </div>
                  <div className="metric-box">
                    <span>Feature dimension</span>
                    <strong>{result.feature_dim}</strong>
                  </div>
                  <div className="metric-box">
                    <span>Vote strength</span>
                    <strong>{voteStrengthPercent}</strong>
                  </div>
                  <div className="metric-box">
                    <span>Model quality blend</span>
                    <strong>{modelQualityPercent}</strong>
                  </div>
                </div>

                <div className="technical-panel">
                  <div className="technical-panel-header">
                    <div>
                      <span className="result-label">Technical readout</span>
                      <h3>Inference record</h3>
                    </div>
                    <div className="tech-badges">
                      <span>{result.input_kind_display || result.input_kind}</span>
                      <span>{result.encoder_label || "Virchow2"}</span>
                      <span>{result.encoder_backbone || result.encoder_type || "local encoder"}</span>
                    </div>
                  </div>

                  <div className="checkpoint-table-wrap">
                    <table className="checkpoint-table">
                      <thead>
                        <tr>
                          <th>Checkpoint</th>
                          <th>Repeat</th>
                          <th>Fold</th>
                          <th>P(MSI-H)</th>
                          <th>Threshold</th>
                          <th>AUROC</th>
                          <th>F1</th>
                          <th>AUPRC</th>
                          <th>Bal Acc</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.per_checkpoint.map((item) => (
                          <tr key={item.checkpoint}>
                            <td>{item.checkpoint.replace("_best_valid.pth", "")}</td>
                            <td>{item.repeat}</td>
                            <td>{item.fold}</td>
                            <td>{(item.probability * 100).toFixed(2)}%</td>
                            <td>{(item.threshold * 100).toFixed(2)}%</td>
                            <td>{item.auroc.toFixed(3)}</td>
                            <td>{item.f1_macro.toFixed(3)}</td>
                            <td>{item.auprc.toFixed(3)}</td>
                            <td>{item.balanced_accuracy.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="explain-panel">
                  <div className="explain-panel-header">
                    <div>
                      <span className="result-label">Interpretation</span>
                      <h3>Technical explanation</h3>
                    </div>
                    <div className="chat-badge">
                      <span>Grounded on current output</span>
                    </div>
                  </div>

                  <div className="explain-list">
                    {technicalNarrative.map((item) => (
                      <div key={item} className="explain-item">
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="result-placeholder-stack viewer-stack-wide">
                <div className="viewer-card">
                  <span>Input</span>
                  <DeepZoomViewer
                    src={specimenViewerSrc}
                    alt="Input specimen viewer"
                    emptyLabel="Upload an image or slide to preview it here."
                    heightClassName="viewer-tall"
                  />
                </div>
                <div className="result-placeholder">
                  <p>Prediction, confidence, runtime details, and tile mapping will appear here.</p>
                </div>
              </div>
            )}
          </section>
        </section>
        ) : null}

        {activeTab === "storage" ? (
          <section className="library-card glass-card">
            <div className="section-top">
              <div>
                <p className="section-kicker">Storage</p>
                <h2>{storageManifest?.title || "Extra SVS library"}</h2>
              </div>
              <span className="state-pill">{storageManifest?.summary.available_total || 0} files</span>
            </div>

            <div className="library-summary-grid">
              <div className="runtime-box">
                <span>Available on VM</span>
                <strong>{storageManifest?.summary.available_total || 0}</strong>
              </div>
              <div className="runtime-box">
                <span>Requested set</span>
                <strong>{storageManifest?.summary.requested_total || 0}</strong>
              </div>
              <div className="runtime-box">
                <span>Ready to test</span>
                <strong>{storageManifest?.summary.available_total || 0}</strong>
              </div>
              <div className="runtime-box">
                <span>Source group</span>
                <strong>Outside trained 200</strong>
              </div>
            </div>

            <p className="runtime-footnote">
              {storageManifest?.summary.note || "These slides are curated outside the preserved 200-slide training cohort and copied to the VM for extra testing."}
            </p>

            <div className="storage-layout">
              <div className="storage-selection-list">
                {storageManifest?.files?.map((item) => {
                  const savedResult = historyLookup.get(item.bucket_name.trim().toLowerCase());
                  const isSelected = activeStorageSample?.bucket_name === item.bucket_name;
                  return (
                    <article
                      key={item.bucket_name}
                      className={`library-item storage-selection-item ${isSelected ? "storage-selection-item-active" : ""}`}
                      onClick={() => setSelectedStorageBucket(item.bucket_name)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          setSelectedStorageBucket(item.bucket_name);
                        }
                      }}
                    >
                      <div className="library-item-top">
                        <div>
                          <h3>{item.patient}</h3>
                        </div>
                        <span className="library-status">
                          {item.available_on_vm ? "Available" : "Pending"}
                        </span>
                      </div>
                      <strong className="library-filename">{item.bucket_name}</strong>
                      <div className="storage-selection-meta">
                        <span>{item.msi_status}</span>
                        <span>{slideTypeForName(item.bucket_name) || "Unknown"}</span>
                        <span>{item.source_group}</span>
                        <span>{savedResult ? "Saved result ready" : "No saved result yet"}</span>
                      </div>
                    </article>
                  );
                })}
              </div>

              {activeStorageSample ? (
                <section className="storage-preview-panel glass-card">
                  <div className="section-top">
                    <div>
                      <p className="section-kicker">Selected SVS</p>
                      <h2>{activeStorageSample.patient}</h2>
                    </div>
                    <span className="state-pill">
                      {activeStorageSample.available_on_vm ? "Available" : "Pending"}
                    </span>
                  </div>

                  <strong className="library-filename">{activeStorageSample.bucket_name}</strong>
                  <div className="storage-selection-meta storage-selection-meta-strong">
                    <span>{activeStorageSample.msi_status}</span>
                    <span>{slideTypeForName(activeStorageSample.bucket_name) || "Unknown"}</span>
                    <span>{activeStorageSample.source_group}</span>
                  </div>

                  <div className="library-actions">
                    <button
                      type="button"
                      className="library-test-button"
                      disabled={!activeStorageSample.available_on_vm || isLoading}
                      onClick={() => void handleStorageTest(activeStorageSample)}
                    >
                      {isLoading ? "Running..." : "Test file"}
                    </button>
                    {historyLookup.get(activeStorageSample.bucket_name.trim().toLowerCase()) ? (
                      <button
                        type="button"
                        className="library-result-button"
                        onClick={() => focusHistoryByUploadedName(activeStorageSample.bucket_name)}
                      >
                        Result available
                      </button>
                    ) : null}
                    {slideTypeForName(activeStorageSample.bucket_name) ? (
                      <button type="button" className="slide-type-pill" disabled>
                        {slideTypeForName(activeStorageSample.bucket_name)}
                      </button>
                    ) : null}
                  </div>

                  <div className="viewer-stack storage-viewer-stack">
                    <div className="viewer-card">
                      <span>Stored specimen</span>
                      <DeepZoomViewer
                        src={activeStorageSample.specimen_preview_data_url}
                        alt={`${activeStorageSample.patient} stored specimen viewer`}
                        emptyLabel="Saved specimen preview is not available yet."
                        heightClassName="viewer-tall"
                      />
                    </div>
                  </div>
                </section>
              ) : null}
            </div>
          </section>
        ) : null}

        {activeTab === "history" ? (
          <section className="library-card glass-card">
            <div className="section-top">
              <div>
                <p className="section-kicker">History</p>
                <h2>Saved prediction results</h2>
              </div>
              <span className="state-pill">{predictionHistory?.count || 0} saved</span>
            </div>

            {predictionHistory?.items?.length ? (
              <div className="history-layout">
                <div className="history-list">
                  {predictionHistory.items.map((item) => (
                    <article
                      key={`${item.saved_at}-${item.uploaded_name}`}
                      id={historyAnchorFor(item)}
                      className={`history-item ${highlightedHistoryKey === historyKeyFor(item) ? "history-item-highlighted" : ""}`}
                      onClick={() => focusHistoryResult(item)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          focusHistoryResult(item);
                        }
                      }}
                    >
                      <div className="history-item-top">
                        <div>
                          <p className="section-kicker">{item.pipeline_mode}</p>
                          <h3>{item.patient || item.uploaded_name}</h3>
                        </div>
                        <div className="history-item-actions">
                          <span className="library-status">{item.confidence_level}</span>
                          <button
                            type="button"
                            className="history-open-button"
                            onClick={(event) => {
                              event.stopPropagation();
                              openHistoryResult(item);
                            }}
                          >
                            Open result
                          </button>
                          <button
                            type="button"
                            className="history-delete-button"
                            disabled={deletingHistoryKey === historyKeyFor(item)}
                            onClick={(event) => {
                              event.stopPropagation();
                              void handleDeleteHistory(item);
                            }}
                          >
                            {deletingHistoryKey === historyKeyFor(item) ? "Deleting..." : "Delete"}
                          </button>
                        </div>
                      </div>
                      <strong className="history-upload-name">{item.uploaded_name}</strong>
                      {slideTypeForName(item.uploaded_name) ? (
                        <div className="slide-type-row">
                          <button type="button" className="slide-type-pill" disabled>
                            {slideTypeForName(item.uploaded_name)}
                          </button>
                        </div>
                      ) : null}
                      <div className="history-summary-grid">
                        <div className="metric-box">
                          <span>Predicted</span>
                          <strong>{item.label}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Expected</span>
                          <strong>{item.expected_label || "-"}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Saved at</span>
                          <strong>{formatSavedAt(item.saved_at)}</strong>
                        </div>
                      </div>
                      <div className="history-metrics">
                        <div className="metric-box">
                          <span>Confidence</span>
                          <strong>{item.confidence_level} {item.confidence_percent.toFixed(1)}%</strong>
                        </div>
                        <div className="metric-box">
                          <span>MSI-H probability</span>
                          <strong>{(item.probability * 100).toFixed(2)}%</strong>
                        </div>
                        <div className="metric-box">
                          <span>Threshold</span>
                          <strong>{(item.threshold * 100).toFixed(2)}%</strong>
                        </div>
                        <div className="metric-box">
                          <span>Tiles used</span>
                          <strong>{item.tile_count}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Checkpoints</span>
                          <strong>{item.checkpoint_count}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Elapsed</span>
                          <strong>{formatEta(item.elapsed_seconds || 0)}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Feature dim</span>
                          <strong>{item.feature_dim}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Encoder</span>
                          <strong>{item.encoder_label || item.encoder_backbone || "-"}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Input kind</span>
                          <strong>{item.input_kind_display || item.input_kind || "-"}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Vote strength</span>
                          <strong>{typeof item.vote_strength_score === "number" ? `${(item.vote_strength_score * 100).toFixed(1)}%` : "-"}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Model quality</span>
                          <strong>{typeof item.model_quality_score === "number" ? `${(item.model_quality_score * 100).toFixed(1)}%` : "-"}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Blend score</span>
                          <strong>{typeof item.confidence_score === "number" ? `${(item.confidence_score * 100).toFixed(1)}%` : "-"}</strong>
                        </div>
                        <div className="metric-box">
                          <span>Comparison</span>
                          <strong>{expectedComparisonLabel(item.expected_label || "", item.label) || "-"}</strong>
                        </div>
                      </div>
                    </article>
                  ))}
                </div>

                {activeHistoryDetail ? (
                  <section className="history-detail glass-card">
                    <div className="section-top">
                      <div>
                        <p className="section-kicker">History detail</p>
                        <h2>{activeHistoryDetail.title}</h2>
                      </div>
                      <span className="state-pill">{activeHistoryDetail.result.confidence_level}</span>
                    </div>

                    <div className="history-detail-actions">
                      <strong>{activeHistoryDetail.item.uploaded_name}</strong>
                      <div className="history-detail-button-row">
                        <button
                          type="button"
                          className="history-open-button"
                          onClick={() => openHistoryResult(activeHistoryDetail.item)}
                        >
                          Open in Predict
                        </button>
                        {slideTypeForName(activeHistoryDetail.item.uploaded_name) ? (
                          <button type="button" className="slide-type-pill" disabled>
                            {slideTypeForName(activeHistoryDetail.item.uploaded_name)}
                          </button>
                        ) : null}
                      </div>
                    </div>

                    <div className="viewer-stack viewer-stack-wide">
                      <div className="viewer-card">
                        <span>Saved input</span>
                        <DeepZoomViewer
                          src={activeHistoryDetail.result.specimen_preview_data_url}
                          alt={`${activeHistoryDetail.title} saved input viewer`}
                          emptyLabel="Saved input preview is not available for this result."
                          heightClassName="viewer-tall"
                        />
                      </div>
                      <div className="viewer-card">
                        <span>Saved output tile map</span>
                        <DeepZoomViewer
                          src={activeHistoryDetail.result.tile_preview_data_url}
                          alt={`${activeHistoryDetail.title} saved output tile map`}
                          emptyLabel="Saved tile map is not available for this result."
                          heightClassName="viewer-grid-height"
                        />
                      </div>
                    </div>

                    <div className="result-hero history-result-hero">
                      <div>
                        <span className="result-label">Prediction</span>
                        <strong>{activeHistoryDetail.result.label}</strong>
                      </div>
                      <div className="confidence-chip">
                        <span>Confidence</span>
                        <strong>{activeHistoryDetail.result.confidence_level}</strong>
                      </div>
                    </div>

                    {activeHistoryDetail.expectedLabel ? (
                      <div className="annotation-summary">
                        <div className="annotation-summary-box">
                          <span>Expected</span>
                          <strong>{activeHistoryDetail.expectedLabel}</strong>
                        </div>
                        <div className={`annotation-summary-box ${activeHistoryDetail.expectedComparison === "Matched expected" ? "match" : "mismatch"}`}>
                          <span>Comparison</span>
                          <strong>{activeHistoryDetail.expectedComparison}</strong>
                        </div>
                      </div>
                    ) : null}

                    <div className="result-metrics">
                      <div className="metric-box">
                        <span>Confidence</span>
                        <strong>{activeHistoryDetail.confidencePercent}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Tiles used</span>
                        <strong>{activeHistoryDetail.result.tile_count}</strong>
                      </div>
                      <div className="metric-box">
                        <span>MSI-H probability</span>
                        <strong>{activeHistoryDetail.probabilityPercent}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Decision threshold</span>
                        <strong>{activeHistoryDetail.thresholdPercent}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Ensemble checkpoints</span>
                        <strong>{activeHistoryDetail.result.checkpoint_count}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Feature dimension</span>
                        <strong>{activeHistoryDetail.result.feature_dim}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Vote strength</span>
                        <strong>{activeHistoryDetail.voteStrengthPercent}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Model quality blend</span>
                        <strong>{activeHistoryDetail.modelQualityPercent}</strong>
                      </div>
                    </div>

                    <div className="technical-panel">
                      <div className="technical-panel-header">
                        <div>
                          <span className="result-label">Saved technical readout</span>
                          <h3>Checkpoint record</h3>
                        </div>
                        <div className="tech-badges">
                          <span>{activeHistoryDetail.result.input_kind_display || activeHistoryDetail.result.input_kind}</span>
                          <span>{activeHistoryDetail.result.encoder_label || "Virchow2"}</span>
                          <span>{activeHistoryDetail.result.encoder_backbone || activeHistoryDetail.result.encoder_type || "local encoder"}</span>
                        </div>
                      </div>

                      <div className="checkpoint-table-wrap">
                        <table className="checkpoint-table">
                          <thead>
                            <tr>
                              <th>Checkpoint</th>
                              <th>Repeat</th>
                              <th>Fold</th>
                              <th>P(MSI-H)</th>
                              <th>Threshold</th>
                              <th>AUROC</th>
                              <th>F1</th>
                              <th>AUPRC</th>
                              <th>Bal Acc</th>
                            </tr>
                          </thead>
                          <tbody>
                            {activeHistoryDetail.result.per_checkpoint.map((checkpoint) => (
                              <tr key={`${checkpoint.checkpoint}-${checkpoint.repeat}-${checkpoint.fold}`}>
                                <td>{checkpoint.checkpoint.replace("_best_valid.pth", "")}</td>
                                <td>{checkpoint.repeat}</td>
                                <td>{checkpoint.fold}</td>
                                <td>{(checkpoint.probability * 100).toFixed(2)}%</td>
                                <td>{(checkpoint.threshold * 100).toFixed(2)}%</td>
                                <td>{checkpoint.auroc.toFixed(3)}</td>
                                <td>{checkpoint.f1_macro.toFixed(3)}</td>
                                <td>{checkpoint.auprc.toFixed(3)}</td>
                                <td>{checkpoint.balanced_accuracy.toFixed(3)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div className="explain-panel">
                      <div className="explain-panel-header">
                        <div>
                          <span className="result-label">Saved interpretation</span>
                          <h3>History explanation</h3>
                        </div>
                        <div className="chat-badge">
                          <span>Grounded on saved output</span>
                        </div>
                      </div>

                      <div className="explain-list">
                        {activeHistoryDetail.technicalNarrative.map((item) => (
                          <div key={item} className="explain-item">
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>
                  </section>
                ) : null}
              </div>
            ) : (
              <div className="result-placeholder history-empty">
                <p>Completed predictions will be saved here automatically after the backend finishes scoring an upload.</p>
              </div>
            )}
          </section>
        ) : null}

        {activeTab === "analysis" ? (
          <section className="library-card glass-card">
            <div className="section-top">
              <div>
                <p className="section-kicker">Analysis</p>
                <h2>Model accuracy and error breakdown</h2>
              </div>
              <span className="state-pill">{analysisSummary ? `${analysisSummary.overview.total_scored} scored` : "Loading"}</span>
            </div>

            {analysisSummary ? (
              <div className="analysis-layout">
                <div className="analysis-overview-grid">
                  <div className="runtime-box">
                    <span>Total tested</span>
                    <strong>{analysisSummary.overview.total_scored}</strong>
                  </div>
                  <div className="runtime-box">
                    <span>Accuracy</span>
                    <strong>{formatAccuracy(analysisSummary.overview.accuracy)}</strong>
                  </div>
                  <div className="runtime-box">
                    <span>False positives</span>
                    <strong>{analysisSummary.overview.false_positive}</strong>
                  </div>
                  <div className="runtime-box">
                    <span>False negatives</span>
                    <strong>{analysisSummary.overview.false_negative}</strong>
                  </div>
                  <div className="runtime-box">
                    <span>Type I error</span>
                    <strong>{analysisSummary.overview.type_i_error}</strong>
                  </div>
                  <div className="runtime-box">
                    <span>Type II error</span>
                    <strong>{analysisSummary.overview.type_ii_error}</strong>
                  </div>
                </div>

                <div className="analysis-chart-grid">
                  <section className="analysis-panel glass-card">
                    <div className="technical-panel-header">
                      <div>
                        <span className="result-label">Confusion</span>
                        <h3>Error types</h3>
                      </div>
                    </div>
                    <div className="analysis-chart">
                      <ResponsiveContainer width="100%" height={260}>
                        <PieChart>
                          <Pie
                            data={[
                              { name: "True MSS", value: analysisSummary.confusion.MSS_to_MSS },
                              { name: "False positive", value: analysisSummary.confusion.MSS_to_MSI_H },
                              { name: "True MSI-H", value: analysisSummary.confusion.MSI_H_to_MSI_H },
                              { name: "False negative", value: analysisSummary.confusion.MSI_H_to_MSS },
                            ]}
                            dataKey="value"
                            nameKey="name"
                            innerRadius={56}
                            outerRadius={92}
                            paddingAngle={3}
                          >
                            {["#4bd5b6", "#ff8a65", "#4f9cff", "#ffce56"].map((fill) => (
                              <Cell key={fill} fill={fill} />
                            ))}
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </section>

                  <section className="analysis-panel glass-card">
                    <div className="technical-panel-header">
                      <div>
                        <span className="result-label">By Batch</span>
                        <h3>Accuracy by source group</h3>
                      </div>
                    </div>
                    <div className="analysis-chart">
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={sourceAccuracyChartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(120,150,172,0.18)" />
                          <XAxis dataKey="name" hide />
                          <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                          <Tooltip formatter={(value) => `${value ?? 0}%`} />
                          <Legend />
                          <Bar dataKey="accuracyPercent" name="Accuracy" fill="#4f9cff" radius={[8, 8, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </section>

                  <section className="analysis-panel glass-card">
                    <div className="technical-panel-header">
                      <div>
                        <span className="result-label">By Slide Type</span>
                        <h3>DX / TS / BS / MS accuracy</h3>
                      </div>
                    </div>
                    <div className="analysis-chart">
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={slideTypeChartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(120,150,172,0.18)" />
                          <XAxis dataKey="name" />
                          <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                          <Tooltip formatter={(value) => `${value ?? 0}%`} />
                          <Legend />
                          <Bar dataKey="accuracyPercent" name="Accuracy" fill="#4bd5b6" radius={[8, 8, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </section>

                  <section className="analysis-panel glass-card">
                    <div className="technical-panel-header">
                      <div>
                        <span className="result-label">Confidence Mix</span>
                        <h3>Prediction confidence buckets</h3>
                      </div>
                    </div>
                    <div className="analysis-chart">
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={analysisSummary.confidence_distribution}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(120,150,172,0.18)" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="count" name="Count" fill="#7b8cff" radius={[8, 8, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </section>
                </div>

                <div className="analysis-table-grid">
                  <section className="analysis-panel glass-card">
                    <div className="technical-panel-header">
                      <div>
                        <span className="result-label">Current model</span>
                        <h3>Serving configuration</h3>
                      </div>
                    </div>
                    <div className="history-metrics">
                      <div className="metric-box">
                        <span>Pipeline</span>
                        <strong>{analysisSummary.current_model.pipeline_mode || "-"}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Approach</span>
                        <strong>{analysisSummary.current_model.approach_label || "-"}</strong>
                      </div>
                      <div className="metric-box">
                        <span>MIL model</span>
                        <strong>{analysisSummary.current_model.mil_model || "-"}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Encoder</span>
                        <strong>{analysisSummary.current_model.encoder_label || "-"}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Feature dim</span>
                        <strong>{analysisSummary.current_model.feature_dim || "-"}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Checkpoints used</span>
                        <strong>{analysisSummary.current_model.selected_checkpoint_count || "-"}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Available checkpoints</span>
                        <strong>{analysisSummary.current_model.available_checkpoints || "-"}</strong>
                      </div>
                      <div className="metric-box">
                        <span>Threshold</span>
                        <strong>{typeof analysisSummary.current_model.mean_threshold === "number" ? formatAccuracy(1 - analysisSummary.current_model.mean_threshold) : "-"}</strong>
                      </div>
                    </div>
                  </section>

                  <section className="analysis-panel glass-card">
                    <div className="technical-panel-header">
                      <div>
                        <span className="result-label">Batch queue</span>
                        <h3>Current and upcoming batches</h3>
                      </div>
                    </div>
                    <div className="checkpoint-table-wrap">
                      <table className="checkpoint-table">
                        <thead>
                          <tr>
                            <th>Batch</th>
                            <th>Status</th>
                            <th>Done</th>
                            <th>Total</th>
                            <th>Autostart</th>
                          </tr>
                        </thead>
                        <tbody>
                          {analysisSummary.batch_progress.map((batch) => (
                            <tr key={batch.batch_name}>
                              <td>{batch.batch_name}</td>
                              <td>{batch.status}</td>
                              <td>{batch.completed_total}</td>
                              <td>{batch.selected_total}</td>
                              <td>{batch.has_autostart ? "Yes" : "No"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </div>

                <section className="analysis-panel glass-card">
                  <div className="technical-panel-header">
                    <div>
                      <span className="result-label">Recent wrong cases</span>
                      <h3>False-positive / false-negative list</h3>
                    </div>
                  </div>
                  <div className="checkpoint-table-wrap">
                    <table className="checkpoint-table">
                      <thead>
                        <tr>
                          <th>Patient</th>
                          <th>Slide</th>
                          <th>Type</th>
                          <th>Expected</th>
                          <th>Predicted</th>
                          <th>P(MSI-H)</th>
                          <th>Confidence</th>
                          <th>Batch</th>
                        </tr>
                      </thead>
                      <tbody>
                        {analysisSummary.recent_wrong_cases.map((item) => (
                          <tr key={`${item.saved_at}-${item.uploaded_name}`}>
                            <td>{item.patient || "-"}</td>
                            <td>{item.uploaded_name || "-"}</td>
                            <td>{item.slide_type || "-"}</td>
                            <td>{item.expected_label || "-"}</td>
                            <td>{item.label || "-"}</td>
                            <td>{typeof item.probability === "number" ? `${(item.probability * 100).toFixed(2)}%` : "-"}</td>
                            <td>{item.confidence_level || "-"}</td>
                            <td>{item.source_group || "-"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              </div>
            ) : (
              <div className="result-placeholder history-empty">
                <p>Analysis is loading from the saved History and current batch runtime.</p>
              </div>
            )}
          </section>
        ) : null}
      </div>

      {isLoading ? (
        <div className="loading-overlay">
          <video
            className="loading-video"
            src="/processing-background.mp4"
            autoPlay
            muted
            loop
            playsInline
          />
          <div className="loading-scrim" />
          <div className="loading-card glass-card">
            <LoaderCircle className="spin" size={26} />
            <h3>Scanning specimen</h3>
            <p>{job?.progress?.detail || liveStageItems.at(-1)?.detail || "Preparing the current inference request."}</p>
            <div className="loading-progress">
              <div className="loading-progress-bar" style={{ width: `${progressPercent}%` }} />
            </div>
            <div className="loading-meta">
              <span>Elapsed {formatEta(elapsedSeconds)}</span>
              <span>Live stage {currentLiveStage}</span>
            </div>
            <div className="loading-stage-list">
              {liveStageItems.map((stage, index) => (
                <div
                  key={stage.stage}
                  className={`loading-stage-item ${index < activeDisplayStageIndex ? "done" : ""} ${index === activeDisplayStageIndex ? "live" : ""}`}
                >
                  <strong>{stage.label}</strong>
                  <span>{stage.detail}</span>
                </div>
              ))}
            </div>
            <p className="loading-note">
              Live progress comes from the backend job stream. Fast mode uses a lighter local pass; exact mode follows the preserved training-matched path.
            </p>
          </div>
        </div>
      ) : null}
    </main>
  );
}
