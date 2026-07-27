import React, { useCallback, useRef } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { toPng } from 'html-to-image';
import { Download } from 'lucide-react';

interface MindMapRendererProps {
  data: any; // JSON from backend
  onNodeClick: (nodeId: string, label: string, context: string) => void;
}

export default function MindMapRenderer({ data, onNodeClick }: MindMapRendererProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [nodes, , onNodesChange] = useNodesState(data.nodes || []);
  const [edges, , onEdgesChange] = useEdgesState(data.edges || []);

  const handleDownload = useCallback(() => {
    if (reactFlowWrapper.current === null) return;
    toPng(reactFlowWrapper.current, { backgroundColor: '#1a1a1a' })
      .then((dataUrl) => {
        const a = document.createElement('a');
        a.setAttribute('download', 'mind-map.png');
        a.setAttribute('href', dataUrl);
        a.click();
      })
      .catch((err) => console.error('Failed to download image', err));
  }, []);

  const handleNodeClick = (_event: React.MouseEvent, node: any) => {
    if (onNodeClick) {
      onNodeClick(node.id, node.data.label, node.data.context || "");
    }
  };

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-color)', background: '#121212' }} ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        fitView
        colorMode="dark"
      >
        <Background />
        <Controls />
        <MiniMap />
        <Panel position="top-right">
          <button onClick={handleDownload} style={{ padding: '8px 12px', borderRadius: '8px', border: 'none', backgroundColor: 'var(--accent-color)', color: '#000', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 600, boxShadow: '0 4px 6px rgba(0,0,0,0.3)' }}>
            <Download size={16} /> Export PNG
          </button>
        </Panel>
      </ReactFlow>
    </div>
  );
}
