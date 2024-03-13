
import React, { useEffect, useRef } from 'react';
export const MatrixEffect = () => {
    const canvasRef = useRef(null);
  
    useEffect(() => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
  
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
  
      const columns = canvas.width / 20;
      let drops = Array.from({ length: columns }).fill(1);
  
      function drawMatrix() {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = '#0F0'; // Green text
        ctx.font = '15pt monospace';
  
        drops.forEach((y, ind) => {
          const text = String.fromCharCode(Math.random() * 128);
          const x = ind * 20;
          ctx.fillText(text, x, y);
          drops[ind] = y > canvas.height || Math.random() > 0.95 ? 0 : y + 20;
        });
      }
  
      let interval = setInterval(drawMatrix, 50);
  
      return () => clearInterval(interval);
    }, []);
  
    return (
      <canvas ref={canvasRef} className="w-full h-full"></canvas>
    );
  };
  