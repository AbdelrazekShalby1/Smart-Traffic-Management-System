import { Navigate, Route, Routes } from "react-router-dom";
import Shell from "./components/layout/Shell";
import LivePage from "./features/live-monitoring/LivePage";
import AssistantPage from "./features/ai-assistant/AssistantPage";
import DashboardPage from "./features/dashboard/DashboardPage";
import ViolationsPage from "./features/violations/ViolationsPage";
export default function App(){return <Routes><Route element={<Shell/>}><Route index element={<Navigate to="/live" replace/>}/><Route path="/live" element={<LivePage/>}/><Route path="/assistant" element={<AssistantPage/>}/><Route path="/dashboard" element={<DashboardPage/>}/><Route path="/violations" element={<ViolationsPage/>}/><Route path="*" element={<Navigate to="/live" replace/>}/></Route></Routes>;}