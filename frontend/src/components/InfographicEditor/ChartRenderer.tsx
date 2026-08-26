import React from 'react';

interface ChartProps {
  type: string;
  data: any;
  color?: string;
}

export const ChartRenderer: React.FC<ChartProps> = ({ type, data, color = "#2563eb" }) => {
  if (!data || !Array.isArray(data)) return null;

  const maxVal = Math.max(...data.map(d => d.value || 0));

  if (type === 'bar') {
    return (
      <div style={{ display: 'flex', alignItems: 'flex-end', height: '150px', gap: '12px', marginTop: '16px' }}>
        {data.map((item, i) => {
          const heightPct = maxVal > 0 ? ((item.value || 0) / maxVal) * 100 : 0;
          return (
            <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1 }}>
              <div style={{ 
                height: `${heightPct}%`, 
                width: '100%', 
                backgroundColor: color, 
                borderRadius: '4px 4px 0 0',
                minHeight: '4px',
                transition: 'height 0.5s ease'
              }} />
              <span style={{ fontSize: '0.75rem', marginTop: '4px', color: '#64748b', textAlign: 'center' }}>
                {item.label}
              </span>
            </div>
          );
        })}
      </div>
    );
  }

  // Fallback for unknown charts
  return (
    <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '8px', border: '1px dashed #cbd5e1' }}>
      [Chart: {type}]
    </div>
  );
};
