import { FormEvent, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Logo } from '../components/Logo'
import { Role, useAuth } from '../store/auth'

const input = 'mt-1.5 block min-h-12 w-full rounded-xl border border-line bg-white px-3 text-base'

export default function Auth({ mode }: { mode: 'login' | 'signup' }) {
  const su = mode === 'signup'
  const { signup, login, demo } = useAuth()
  const nav = useNavigate()
  const [err, setErr] = useState('')
  const [show, setShow] = useState(false)

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = new FormData(e.currentTarget)
    const email = String(f.get('email')).trim(), pw = String(f.get('pw')), name = String(f.get('name') ?? '').trim()
    if (!email.includes('@') || !email.split('@')[1]?.includes('.')) return setErr('Enter a valid email.')
    if (pw.length < 8) return setErr('Password needs at least 8 characters.')
    if (su) {
      if (name.length < 2) return setErr('Enter your full name.')
      if (pw !== f.get('pw2')) return setErr('Passwords do not match.')
      await signup(name, email, pw, f.get('role') as Role)
    } else {
      const m = await login(email, pw)
      if (m) return setErr(m)
    }
    nav('/dashboard')
  }

  const pt = show ? 'text' : 'password'
  return (
    <div className="grid min-h-screen place-items-center p-5">
      <div className="w-full max-w-md rounded-3xl border border-line bg-white p-7 shadow-sm">
        <Link to="/"><Logo className="mx-auto h-16" /></Link>
        <h1 className="mt-4 mb-1 text-2xl font-extrabold">{su ? 'Create your account' : 'Welcome back'}</h1>
        <p className="mt-0 mb-4 text-sm text-mute">Authentication is simulated and stays in this browser.</p>
        <form onSubmit={submit} noValidate>
          {su && <label className="mb-3 block text-sm font-semibold">Full name<input name="name" className={input} autoComplete="name" /></label>}
          <label className="mb-3 block text-sm font-semibold">Email<input name="email" type="email" className={input} autoComplete="email" /></label>
          <label className="mb-3 block text-sm font-semibold">Password<input name="pw" type={pt} className={input} /></label>
          {su && <>
            <label className="mb-3 block text-sm font-semibold">Confirm password<input name="pw2" type={pt} className={input} /></label>
            <label className="mb-3 block text-sm font-semibold">I am a<select name="role" className={input}><option>Patient</option><option>Caregiver</option><option>Doctor</option></select></label>
          </>}
          <p role="alert" className="mb-2 min-h-5 text-sm text-red-700">{err}</p>
          <button className="min-h-12 w-full rounded-xl bg-forest font-semibold text-white" type="submit">{su ? 'Sign up' : 'Log in'}</button>
        </form>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-2 text-sm">
          <label className="flex items-center gap-2"><input type="checkbox" onChange={(e) => setShow(e.target.checked)} /> Show password</label>
          <Link className="font-semibold text-teal" to={su ? '/login' : '/signup'}>{su ? 'Have an account? Log in' : 'Create account'}</Link>
        </div>
        {!su && <button type="button" onClick={() => { demo(); nav('/dashboard') }} className="mt-4 min-h-11 w-full rounded-xl border border-forest font-semibold text-forest">Continue as demo user</button>}
      </div>
    </div>
  )
}
