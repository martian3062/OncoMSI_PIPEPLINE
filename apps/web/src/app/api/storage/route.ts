const DJANGO_API_BASE_URL =
  process.env.DJANGO_API_BASE_URL || "http://127.0.0.1:8000";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const url = new URL(`${DJANGO_API_BASE_URL}/api/storage-samples/`);
  for (const key of ["compact", "bucket_name"]) {
    const value = searchParams.get(key);
    if (value) {
      url.searchParams.set(key, value);
    }
  }
  const response = await fetch(url.toString(), {
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
