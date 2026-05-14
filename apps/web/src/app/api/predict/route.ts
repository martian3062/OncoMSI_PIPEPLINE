import { proxyJson } from "../_lib/proxy";

const DJANGO_API_BASE_URL =
  process.env.DJANGO_API_BASE_URL || "http://127.0.0.1:8000";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const jobId = searchParams.get("job_id");
  const mode = searchParams.get("mode");
  const target = jobId
    ? `${DJANGO_API_BASE_URL}/api/predict-jobs/${encodeURIComponent(jobId)}/`
    : `${DJANGO_API_BASE_URL}/api/predict-metadata/`;
  const url = new URL(target);
  if (mode) {
    url.searchParams.set("mode", mode);
  }
  return proxyJson(url.toString());
}

export async function POST(request: Request) {
  const { searchParams } = new URL(request.url);
  const mode = searchParams.get("mode");
  const url = new URL(`${DJANGO_API_BASE_URL}/api/predict-jobs/`);
  if (mode) {
    url.searchParams.set("mode", mode);
  }
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers.set("content-type", contentType);
  }
  const init: RequestInit & { duplex: "half" } = {
    method: "POST",
    headers,
    body: request.body,
    cache: "no-store",
    duplex: "half",
  };
  return proxyJson(url.toString(), init);
}
