import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, Circle, Loader2, AlertCircle } from 'lucide-react';
import './GenerationProgress.css';

interface ProgressStep {
  id: string;
  label: string;
  status: 'pending' | 'loading' | 'completed' | 'error';
  detail?: string;
}

interface Props {
  steps: ProgressStep[];
  overallProgress?: number; // 0-100
}

export const GenerationProgress: React.FC<Props> = ({ steps, overallProgress }) => {
  return (
    <div className="generation-progress-container">
      <h3>Generating Infographic...</h3>
      {overallProgress !== undefined && (
        <div className="progress-bar-bg">
          <motion.div 
            className="progress-bar-fill"
            initial={{ width: 0 }}
            animate={{ width: `${overallProgress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      )}
      
      <div className="steps-list">
        {steps.map((step, index) => (
          <div key={step.id} className={`step-item ${step.status}`}>
            <div className="step-icon">
              {step.status === 'completed' && <CheckCircle size={20} className="icon-success" />}
              {step.status === 'loading' && <Loader2 size={20} className="icon-loading animate-spin" />}
              {step.status === 'pending' && <Circle size={20} className="icon-pending" />}
              {step.status === 'error' && <AlertCircle size={20} className="icon-error" />}
            </div>
            <div className="step-content">
              <span className="step-label">{step.label}</span>
              {step.detail && <span className="step-detail">{step.detail}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
