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
  const response = await fetch(url.toString(), { cache: "no-store" });
  const payload = await response.text();
  return new Response(payload, {
    status: response.status,
    headers: {
      "content-type": response.headers.get("content-type") || "application/json",
    },
  });
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
  const response = await fetch(url.toString(), init);
  const payload = await response.text();
  return new Response(payload, {
    status: response.status,
    headers: {
      "content-type": response.headers.get("content-type") || "application/json",
    },
  });
}
