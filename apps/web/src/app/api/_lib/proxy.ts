type ProxyOptions = {
  method?: string;
  headers?: HeadersInit;
  body?: BodyInit | null;
};

function jsonHeaders(contentType?: string | null) {
  return {
    "content-type": contentType || "application/json",
  };
}

export async function proxyJson(url: string, options: ProxyOptions = {}) {
  try {
    const response = await fetch(url, {
      cache: "no-store",
      ...options,
    });
    const payload = await response.text();
    return new Response(payload, {
      status: response.status,
      headers: jsonHeaders(response.headers.get("content-type")),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Proxy request failed.";
    return Response.json(
      {
        error: message,
        backend_available: false,
      },
      {
        status: 503,
      },
    );
  }
}
