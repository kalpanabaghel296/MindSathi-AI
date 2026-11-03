# MindCare AI

Frontend for **AI-Based Early Detection and Support System for Depression Using Behavioral and Wearable Data**.
A research prototype. It shows risk indicators, never diagnoses.

## Stack
React 18, TypeScript, Vite, Tailwind CSS v4, React Router, Zustand, Recharts, Lucide.

## Run locally (Windows PowerShell)
```powershell
npm install
npm run dev        # opens at http://localhost:5173
npm run build      # type-check + production build
```
Requires Node.js 18 or newer (check with `node --version`).

## What is implemented
- Landing page, login, signup, protected routes, logout (prototype auth, see below)
- Dashboard: Overview (risk + sleep charts), Explainable AI (SHAP bars), Mood tracking (saved in the browser)
- Service layer in `src/services/api.ts` and demo data in `src/data/mock.ts`
- Logo integration (`public/logo`, favicon)

## What is NOT implemented yet
Risk monitoring, behavioral analytics, companion, doctor and caregiver dashboards, health records (EHR/FHIR),
privacy center, wellness goals, notifications, data upload, onboarding, settings. These exist in the
single-file prototype and are being ported.

## Honest limitations
- **Auth is a browser-only prototype.** The account (with a SHA-256 hash, not a password) is kept in localStorage. It is not secure for production. Replace `src/store/auth.ts` with Supabase, Firebase or your backend.
- **All numbers are demo values**, except the SHAP bars and the Low/Moderate/High cut-offs (27% and 42%), which come from the project's own model run. Nothing here is a medical diagnosis.
- No ML model, LLM, wearable or hospital system is connected.

## Connect your Python backend
1. Copy `.env.example` to `.env` and set `VITE_API_BASE_URL=http://127.0.0.1:5000`.
2. Expose these endpoints (JSON): `/api/risk/history` (number[]), `/api/analytics/sleep` (`{day, hours}[]`), `/api/explain/global` (`{name, value, context}[]`).
3. Without the variable, the app uses the demo data. Never put API keys in frontend code.

## Folder structure
```
public/logo        logo files
src/components     Layout, Logo
src/pages          Landing, Auth, Overview, Explain, Mood
src/services       api.ts (backend integration point)
src/store          auth.ts, mood.ts (Zustand)
src/data           mock.ts (demo data)
```

## Push to GitHub (PowerShell)
Create an empty repository on github.com first (no README), then:
```powershell
git init
git add .
git commit -m "MindCare AI React frontend"
git branch -M main
git remote add origin https://github.com/<your-username>/mindcare-ai.git
git push -u origin main
```
