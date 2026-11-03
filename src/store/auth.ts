import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Role = 'Patient' | 'Caregiver' | 'Doctor'
interface User { name: string; email: string; role: Role; hash: string }
interface AuthState {
  user: User | null
  session: boolean
  signup: (name: string, email: string, pw: string, role: Role) => Promise<void>
  login: (email: string, pw: string) => Promise<string | null>
  demo: () => void
  logout: () => void
}

async function sha(t: string) {
  const b = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(t))
  return [...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, '0')).join('')
}

// PROTOTYPE auth: account lives in this browser only. Replace with Supabase/Firebase/your backend.
export const useAuth = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      session: false,
      signup: async (name, email, pw, role) => {
        const e = email.toLowerCase()
        set({ user: { name, email: e, role, hash: await sha(e + pw) }, session: true })
      },
      login: async (email, pw) => {
        const u = get().user, e = email.toLowerCase()
        if (!u || u.email !== e || u.hash !== (await sha(e + pw))) return 'Email or password is incorrect, or no account exists in this browser.'
        set({ session: true })
        return null
      },
      demo: () => set((s) => ({ user: s.user ?? { name: 'Demo User', email: 'demo@mindcare.test', role: 'Patient', hash: '' }, session: true })),
      logout: () => set({ session: false }),
    }),
    { name: 'mindcare-auth' },
  ),
)
