import { proxyJson } from "../_lib/proxy";

const DJANGO_API_BASE_URL =
  process.env.DJANGO_API_BASE_URL || "http://127.0.0.1:8000";

export async function POST(request: Request) {
  const { searchParams } = new URL(request.url);
  const mode = searchParams.get("mode");
  const url = new URL(`${DJANGO_API_BASE_URL}/api/storage-samples/test/`);
  if (mode) {
    url.searchParams.set("mode", mode);
  }
  const formData = await request.formData();
  const bucketName = String(formData.get("bucket_name") || "").trim();
  const body = new URLSearchParams({ bucket_name: bucketName }).toString();
  const init: RequestInit = {
    method: "POST",
    headers: {
      "content-type": "application/x-www-form-urlencoded",
    },
    body,
    cache: "no-store",
  };
  return proxyJson(url.toString(), init);
}
