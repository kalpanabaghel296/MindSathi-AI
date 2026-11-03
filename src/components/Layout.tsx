import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { LayoutDashboard, Search, Smile, LogOut } from 'lucide-react'
import { Logo } from './Logo'
import { useAuth } from '../store/auth'

const links = [
  { to: '/dashboard', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/dashboard/explain', label: 'Explainable AI', icon: Search },
  { to: '/dashboard/mood', label: 'Mood tracking', icon: Smile },
]

export default function Layout() {
  const { user, logout } = useAuth()
  const nav = useNavigate()
  return (
    <div className="min-h-screen md:grid md:grid-cols-[250px_1fr]">
      <aside className="border-b border-line bg-white p-3 md:sticky md:top-0 md:h-screen md:border-r md:border-b-0 md:p-5">
        <Logo variant="icon" className="mb-3 hidden h-16 md:block" />
        <nav className="flex gap-1 overflow-x-auto md:flex-col" aria-label="Dashboard">
          {links.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => `flex min-h-11 items-center gap-2 whitespace-nowrap rounded-xl px-3 text-[15px] font-medium ${isActive ? 'bg-mint font-bold text-forest' : 'text-mute hover:bg-cream'}`}>
              <Icon size={18} aria-hidden /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-4 hidden border-t border-line pt-4 text-sm md:block">
          <p className="m-0 font-semibold">{user?.name}</p>
          <p className="m-0 mb-3 text-mute">{user?.role}</p>
          <button onClick={() => { logout(); nav('/') }} className="flex min-h-11 items-center gap-2 rounded-xl border border-forest px-4 font-semibold text-forest"><LogOut size={16} /> Log out</button>
        </div>
      </aside>
      <main className="min-w-0 p-5 md:p-8"><Outlet /></main>
    </div>
  )
}
