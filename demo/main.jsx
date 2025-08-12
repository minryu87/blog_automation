import React from 'react';
import { createRoot } from 'react-dom/client';
import HospitalMarketingDashboard from './HospitalMarketingDashboard.jsx';

const container = document.getElementById('root');
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <HospitalMarketingDashboard />
  </React.StrictMode>
);
