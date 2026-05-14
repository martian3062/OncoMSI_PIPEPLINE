const DJANGO_API_BASE_URL =
  process.env.DJANGO_API_BASE_URL || "http://127.0.0.1:8000";

export async function GET() {
  const response = await fetch(`${DJANGO_API_BASE_URL}/api/batch-status/`, {
    cache: "no-store",
  });
  const payload = await response.text();
  return new Response(payload, {
    status: response.status,
    headers: {
      "content-type": response.headers.get("content-type") || "application/json",
    },
  });
}
