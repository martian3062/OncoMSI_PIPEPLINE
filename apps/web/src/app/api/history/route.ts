import { proxyJson } from "../_lib/proxy";

const DJANGO_API_BASE_URL =
  process.env.DJANGO_API_BASE_URL || "http://127.0.0.1:8000";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const url = new URL(`${DJANGO_API_BASE_URL}/api/prediction-history/`);
  for (const key of ["limit", "compact", "job_id", "saved_at", "uploaded_name"]) {
    const value = searchParams.get(key);
    if (value) {
      url.searchParams.set(key, value);
    }
  }
  return proxyJson(url.toString());
}

export async function DELETE(request: Request) {
  const url = new URL(`${DJANGO_API_BASE_URL}/api/prediction-history/`);
  return proxyJson(url.toString(), {
    method: "DELETE",
    headers: {
      "content-type": "application/json",
    },
    body: await request.text(),
  });
}
