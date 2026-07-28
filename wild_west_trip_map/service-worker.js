const CACHE_NAME = "wild-west-trip-v1";
const RUNTIME_CACHE_NAME = "wild-west-runtime-v1";

const APP_SHELL_FILES = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./tripData.js",
  "./manifest.webmanifest",
  "./icon.svg"
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(APP_SHELL_FILES);
    })
  );

  self.skipWaiting();
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames
          .filter(cacheName => {
            return cacheName !== CACHE_NAME && cacheName !== RUNTIME_CACHE_NAME;
          })
          .map(cacheName => caches.delete(cacheName))
      );
    })
  );

  self.clients.claim();
});

self.addEventListener("fetch", event => {
  if (event.request.method !== "GET") {
    return;
  }

  const requestUrl = new URL(event.request.url);

  const isSameOrigin = requestUrl.origin === self.location.origin;
  const isLeafletFile =
    requestUrl.hostname === "unpkg.com" &&
    requestUrl.pathname.includes("leaflet@1.9.4");

  const isOpenStreetMapTile =
    requestUrl.hostname.endsWith("tile.openstreetmap.org");

  if (isOpenStreetMapTile) {
    return;
  }

  if (!isSameOrigin && !isLeafletFile) {
    return;
  }

  event.respondWith(
    caches.match(event.request).then(cachedResponse => {
      if (cachedResponse) {
        return cachedResponse;
      }

      return fetch(event.request)
        .then(networkResponse => {
          if (
            networkResponse &&
            (networkResponse.ok || networkResponse.type === "opaque")
          ) {
            const responseClone = networkResponse.clone();
            const cacheToUse = isSameOrigin ? CACHE_NAME : RUNTIME_CACHE_NAME;

            caches.open(cacheToUse).then(cache => {
              cache.put(event.request, responseClone);
            });
          }

          return networkResponse;
        })
        .catch(() => {
          if (event.request.mode === "navigate") {
            return caches.match("./index.html");
          }
        });
    })
  );
});