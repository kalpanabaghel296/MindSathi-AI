import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Landing from './pages/Landing'
import Auth from './pages/Auth'
import Overview from './pages/Overview'
import Explain from './pages/Explain'
import Mood from './pages/Mood'
import { useAuth } from './store/auth'

function Protected() {
  const session = useAuth((s) => s.session)
  return session ? <Layout /> : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Auth mode="login" />} />
      <Route path="/signup" element={<Auth mode="signup" />} />
      <Route path="/dashboard" element={<Protected />}>
        <Route index element={<Overview />} />
        <Route path="explain" element={<Explain />} />
        <Route path="mood" element={<Mood />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
