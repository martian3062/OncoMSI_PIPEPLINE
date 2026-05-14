import { proxyJson } from "../_lib/proxy";

const DJANGO_API_BASE_URL =
  process.env.DJANGO_API_BASE_URL || "http://127.0.0.1:8000";

export async function GET() {
  return proxyJson(`${DJANGO_API_BASE_URL}/api/batch-status/`);
}
