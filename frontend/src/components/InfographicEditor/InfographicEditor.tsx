import React, { useState } from 'react';
import { jsPDF } from 'jspdf';
import { toPng, toJpeg, toSvg } from 'html-to-image';
import { Download, Play, Edit3, Image as ImageIcon, Loader2 } from 'lucide-react';
import { InfographicRenderer } from './InfographicRenderer';
import { GenerationProgress } from './GenerationProgress';
import './InfographicEditor.css';

interface Props {
  backendUrl: string;
}

export const InfographicEditor: React.FC<Props> = ({ backendUrl }) => {
  const [topic, setTopic] = useState('');
  const [content, setContent] = useState('');
  const [style, setStyle] = useState('Modern');
  const [isGenerating, setIsGenerating] = useState(false);
  const [infographicData, setInfographicData] = useState<any>(null);
  
  const [progressSteps, setProgressSteps] = useState<{id: string, label: string, status: 'pending' | 'loading' | 'completed' | 'error', detail?: string}[]>([
    { id: 'analyze', label: 'Analyzing content', status: 'pending' },
    { id: 'design', label: 'Designing layout', status: 'pending' },
    { id: 'visuals', label: 'Generating AI visuals', status: 'pending' },
  ]);
  const [overallProgress, setOverallProgress] = useState(0);

  const updateStep = (id: string, status: any, detail?: string) => {
    setProgressSteps(prev => prev.map(s => s.id === id ? { ...s, status, detail } : s));
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    setInfographicData(null);
    setOverallProgress(10);
    
    // Reset steps
    setProgressSteps([
      { id: 'analyze', label: 'Analyzing content', status: 'loading' },
      { id: 'design', label: 'Designing layout', status: 'pending' },
      { id: 'visuals', label: 'Generating AI visuals', status: 'pending' },
    ]);

    try {
      // 1. Get structured JSON from StudySnapAI
      const res = await fetch(`${backendUrl}/api/infographic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          custom_prompt: `Topic: ${topic}\nContent: ${content}`,
          style: style,
          detail_level: "Standard"
        })
      });
      
      const data = await res.json();
      if (!data.success || !data.data) throw new Error("Failed to generate structure");
      
      const layoutData = data.data;
      updateStep('analyze', 'completed');
      updateStep('design', 'completed');
      setOverallProgress(50);
      
      // Mount layout to trigger visual rendering
      setInfographicData(layoutData);
      
      // 2. Generate visuals via ComfyUI
      updateStep('visuals', 'loading', 'Queueing prompts to ComfyUI...');
      
      const updatedSections = [...layoutData.sections];
      let totalVisuals = updatedSections.filter(s => s.type === 'illustration' && s.visualAssetPrompt).length;
      let completedVisuals = 0;

      for (let i = 0; i < updatedSections.length; i++) {
        const sec = updatedSections[i];
        if (sec.type === 'illustration' && sec.visualAssetPrompt) {
          try {
            const imgRes = await fetch(`${backendUrl}/api/generate-image`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ prompt: sec.visualAssetPrompt })
            });
            const imgData = await imgRes.json();
            if (imgData.success) {
              updatedSections[i].imageUrl = imgData.filename;
            }
          } catch (err) {
            console.error("Image generation failed for section", sec.id, err);
          }
          completedVisuals++;
          setOverallProgress(50 + (completedVisuals / totalVisuals) * 50);
          
          // Re-trigger render
          setInfographicData({ ...layoutData, sections: updatedSections });
        }
      }

      updateStep('visuals', 'completed', 'All images generated!');
      setOverallProgress(100);

    } catch (err) {
      console.error(err);
      updateStep('analyze', 'error', 'Failed to communicate with AI');
    } finally {
      setTimeout(() => setIsGenerating(false), 2000);
    }
  };

  const handleExport = async (format: 'png' | 'jpg' | 'svg' | 'pdf') => {
    const node = document.getElementById('infographic-export-node');
    if (!node) return;

    try {
      if (format === 'png') {
        const dataUrl = await toPng(node);
        downloadFile(dataUrl, 'infographic.png');
      } else if (format === 'jpg') {
        const dataUrl = await toJpeg(node, { quality: 0.95 });
        downloadFile(dataUrl, 'infographic.jpg');
      } else if (format === 'svg') {
        const dataUrl = await toSvg(node);
        downloadFile(dataUrl, 'infographic.svg');
      } else if (format === 'pdf') {
        const dataUrl = await toPng(node);
        const pdf = new jsPDF({
          orientation: 'portrait',
          unit: 'px',
          format: [node.offsetWidth, node.offsetHeight]
        });
        pdf.addImage(dataUrl, 'PNG', 0, 0, node.offsetWidth, node.offsetHeight);
        pdf.save('infographic.pdf');
      }
    } catch (err) {
      console.error("Export failed", err);
      alert("Failed to export infographic.");
    }
  };

  const downloadFile = (dataUrl: string, filename: string) => {
    const link = document.createElement('a');
    link.download = filename;
    link.href = dataUrl;
    link.click();
  };

  const handleSectionUpdate = (id: string, newContent: string) => {
    setInfographicData((prev: any) => {
      if (!prev) return prev;
      return {
        ...prev,
        sections: prev.sections.map((s: any) => s.id === id ? { ...s, content: newContent } : s)
      };
    });
  };

  return (
    <div className="infographic-editor-layout">
      {/* LEFT PANEL: Controls */}
      <div className="editor-sidebar left-sidebar">
        <h2 className="sidebar-title"><Edit3 size={20} /> Content</h2>
        
        <div className="form-group">
          <label>Topic</label>
          <input 
            type="text" 
            placeholder="e.g. History of Rome" 
            value={topic} 
            onChange={e => setTopic(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Study Material</label>
          <textarea 
            placeholder="Paste your notes here..." 
            rows={8}
            value={content}
            onChange={e => setContent(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Style Theme</label>
          <select value={style} onChange={e => setStyle(e.target.value)}>
            <option>Modern</option>
            <option>Dark</option>
            <option>Minimal</option>
            <option>Corporate</option>
          </select>
        </div>

        <button 
          className="generate-btn" 
          onClick={handleGenerate}
          disabled={isGenerating || !topic}
        >
          {isGenerating ? <Loader2 className="animate-spin" /> : <Play size={18} />}
          {isGenerating ? 'Generating...' : 'Create Infographic'}
        </button>
      </div>

      {/* CENTER PANEL: Canvas */}
      <div className="editor-canvas-area">
        {isGenerating && !infographicData && (
          <div className="generation-overlay">
            <GenerationProgress steps={progressSteps} overallProgress={overallProgress} />
          </div>
        )}
        
        <div className="canvas-wrapper">
          {infographicData ? (
            <InfographicRenderer 
              data={infographicData} 
              onUpdateSection={handleSectionUpdate} 
              backendUrl={backendUrl}
            />
          ) : !isGenerating ? (
            <div className="empty-canvas">
              <ImageIcon size={48} className="empty-icon" />
              <p>Enter a topic and generate your infographic</p>
            </div>
          ) : null}
        </div>
      </div>

      {/* RIGHT PANEL: Properties & Export */}
      <div className="editor-sidebar right-sidebar">
        <h2 className="sidebar-title"><Download size={20} /> Export</h2>
        
        <div className="export-actions">
          <button onClick={() => handleExport('png')}>Export as PNG</button>
          <button onClick={() => handleExport('jpg')}>Export as JPG</button>
          <button onClick={() => handleExport('svg')}>Export as SVG</button>
          <button onClick={() => handleExport('pdf')} className="primary-export">Export as PDF</button>
        </div>
      </div>
    </div>
  );
};
