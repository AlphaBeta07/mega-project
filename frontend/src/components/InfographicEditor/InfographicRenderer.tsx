import React from 'react';
import { ChartRenderer } from './ChartRenderer';
import { Loader2 } from 'lucide-react';
import './InfographicRenderer.css';

interface Section {
  id: string;
  title: string;
  content: string;
  type: string;
  visualAssetPrompt?: string;
  imageUrl?: string;
  chartData?: any;
}

interface InfographicData {
  title: string;
  subtitle: string;
  theme: string;
  sections: Section[];
}

interface Props {
  data: InfographicData | null;
  onUpdateSection?: (id: string, newContent: string) => void;
  backendUrl: string;
}

export const InfographicRenderer: React.FC<Props> = ({ data, onUpdateSection, backendUrl }) => {
  if (!data) {
    return <div className="infographic-placeholder">Waiting for infographic data...</div>;
  }

  const handleTextEdit = (id: string, e: React.FormEvent<HTMLDivElement>) => {
    if (onUpdateSection) {
      onUpdateSection(id, e.currentTarget.textContent || '');
    }
  };

  return (
    <div className={`infographic-canvas theme-${data.theme.toLowerCase().replace(/\s+/g, '-')}`} id="infographic-export-node">
      <header className="infographic-header">
        <h1 contentEditable suppressContentEditableWarning className="editable-text">{data.title}</h1>
        <p contentEditable suppressContentEditableWarning className="editable-text">{data.subtitle}</p>
      </header>

      <div className="infographic-body">
        {data.sections.map((section) => (
          <div key={section.id} className={`infographic-section type-${section.type}`}>
            
            {section.type === 'illustration' && (
              <div className="section-visual">
                {section.imageUrl ? (
                  <img src={`${backendUrl}/api/image/${section.imageUrl}`} alt={section.title} crossOrigin="anonymous" />
                ) : (
                  <div className="visual-placeholder">
                    <Loader2 className="animate-spin" />
                    <span>Generating visual...</span>
                  </div>
                )}
              </div>
            )}

            <div className="section-content">
              <h3 contentEditable suppressContentEditableWarning className="editable-text">{section.title}</h3>
              <div 
                className="editable-text content-body" 
                contentEditable 
                suppressContentEditableWarning
                onBlur={(e) => handleTextEdit(section.id, e)}
              >
                {section.content}
              </div>
              
              {section.type === 'chart' && section.chartData && (
                <ChartRenderer type="bar" data={section.chartData} />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
