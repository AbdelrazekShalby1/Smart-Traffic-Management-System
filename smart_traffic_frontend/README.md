# Smart Traffic Intelligence — Frontend

React + TypeScript + Vite frontend with linked routes:
- `/live` — Live Monitoring
- `/assistant` — AI Traffic Assistant
- `/dashboard` — Traffic Analytics
- `/violations` — Violation Management

The sidebar connects the pages. Data is fetched from the backend; the app does not create fake traffic metrics.

## Run

Requires Node.js 20+ and npm.

```bash
npm install
```

Copy `.env.example` to `.env` and set `VITE_API_BASE_URL` and optionally `VITE_CAMERA_STREAM_URL`.

```bash
npm run dev
```

Open `http://localhost:5173`.

## Expected backend endpoints

- `GET /api/health`
- `GET /api/traffic/live`
- `GET /api/traffic/vehicles`
- `GET /api/traffic/density`
- `GET /api/traffic/events`
- `GET /api/violations`
- `GET /api/dashboard/summary`
- `POST /api/agent/chat` with `{"message":"...","conversation_id":"optional"}`
- `GET /api/camera/stream` returning an MJPEG stream if using the default stream URL

The UI expects live metrics such as `active_vehicles`, `vehicle_counts`, `average_speed`, `speeding_vehicles`, `emergency_vehicles`, `density`, and `fps`. The chat response should contain `answer` (or `response`/`message`) and may include `tool_calls`.

The frontend is implemented, but live camera/metrics/chat require the matching backend endpoints to exist. Adapt `src/services/api.ts` to match your actual API contract.
