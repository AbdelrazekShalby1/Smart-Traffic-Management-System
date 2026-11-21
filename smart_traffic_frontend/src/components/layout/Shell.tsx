import { NavLink, Outlet, useLocation } from "react-router-dom";
import { Activity, Bot, ChartNoAxesCombined, ChevronDown, CircleHelp, Command, RadioTower, Settings2, ShieldAlert, Signal } from "lucide-react";
const nav = [
  { to: "/live", label: "Live Monitoring", icon: RadioTower },
  { to: "/assistant", label: "AI Assistant", icon: Bot },
  { to: "/dashboard", label: "Analytics Dashboard", icon: ChartNoAxesCombined },
  { to: "/violations", label: "Violations", icon: ShieldAlert },
];
const titles: Record<string,string> = { "/live":"Live Monitoring", "/assistant":"AI Traffic Assistant", "/dashboard":"Traffic Analytics", "/violations":"Violation Management" };
export default function Shell() {
  const location = useLocation(); const title = titles[location.pathname] || "Smart Traffic Intelligence";
  return <div className="app-shell"><aside className="sidebar">
    <NavLink to="/live" className="brand"><span className="brand-mark"><Activity size={22}/></span><span><strong>TRAFFIC<span className="brand-accent">IQ</span></strong><small>INTELLIGENCE PLATFORM</small></span></NavLink>
    <div className="workspace-label">WORKSPACE</div><nav className="side-nav">{nav.map(({to,label,icon:Icon})=><NavLink key={to} to={to} className={({isActive})=>`nav-link ${isActive?"active":""}`}><Icon size={18}/><span>{label}</span>{to==="/live"&&<span className="nav-live-dot"/>}</NavLink>)}</nav>
    <div className="sidebar-bottom"><div className="system-card"><div className="system-card-head"><span className="status-dot"/> SYSTEM STATUS</div><strong>Connect your backend</strong><p>Live metrics appear when the API is available.</p></div>
      <NavLink to="/dashboard" className="nav-link"><Settings2 size={18}/><span>System overview</span></NavLink><a className="nav-link" href="https://fastapi.tiangolo.com/" target="_blank" rel="noreferrer"><CircleHelp size={18}/><span>API integration help</span></a>
      <div className="profile"><div className="avatar">ST</div><div className="profile-copy"><strong>Traffic Operator</strong><small>Monitoring workspace</small></div><ChevronDown size={15}/></div>
    </div></aside><main className="main-area"><header className="topbar"><div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><strong>{title}</strong></div><div className="topbar-actions"><span className="environment-tag"><span className="status-dot"/> LOCAL ENVIRONMENT</span><span className="top-icon"><Signal size={17}/></span><span className="top-icon"><Command size={17}/></span></div></header><div className="page-content"><Outlet/></div><footer className="footer"><span><Activity size={13}/> SMART TRAFFIC INTELLIGENCE</span><span>Live data depends on backend connectivity.</span></footer></main></div>;
}