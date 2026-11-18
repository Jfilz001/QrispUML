import React, { useEffect, useRef, useState } from 'react';
import $ from 'jquery';
import _ from 'lodash';
import Backbone from 'backbone';
import * as joint from 'jointjs';
import 'jointjs/dist/joint.css';

// Importiere die UML-Datenbank
import umlData from './qrisp_uml_data.json';

// Globals für JointJS
if (typeof window !== 'undefined') {
  window.$ = $;
  window.jQuery = $;
  window._ = _;
  window.Backbone = Backbone;
}

// Definition der verfügbaren Kategorien
// Überarbeitete Liste der verfügbaren Kategorien
const CATEGORIES = [
  'Algorithms & Problems',
  'Quantumenvironments & Quantumprimitives',
  'Operations',
  'QuantumVariables',
  'Hamiltonian',
  'Tensors',
  'Backends',
  'Metrics',
  'Permeability',
  'QuantumManagement',
  'Other'
];

const getCategory = (cls) => {
  const fp = (cls.file_path || '').toLowerCase();
  const name = (cls.class_name || '').toLowerCase();

  // 1. High-Level Algorithmen & Problem-Solver (Sehr spezifisch)
  if (
    fp.includes('algorithm') || 
    fp.includes('problem') || 
    fp.includes('vqe') || 
    fp.includes('qaoa') || 
    fp.includes('backtracking') ||
    fp.includes('shor') ||
    fp.includes('grover') ||
    name.includes('vqe') ||
    name.includes('qaoa')
  ) {
    return 'Algorithms & Problems';
  }

  // 2. Quantum Environments & Primitives (Primitiven wie Phase Estimation, QFT, etc.)
  if (
    fp.includes('environment') || 
    fp.includes('primitive') || 
    fp.includes('alg_primitives') ||
    name.includes('environment') ||
    name.includes('qft') ||
    name.includes('phaseestimation')
  ) {
    return 'Quantumenvironments & Quantumprimitives';
  }

  // 3. Quanten-Operationen & Gates
  if (
    fp.includes('operation') || 
    fp.includes('mcx') || 
    fp.includes('gate') || 
    name.includes('gate') || 
    name.includes('operation')
  ) {
    return 'Operations';
  }

  // 4. Physikalische Modelle & Hamiltonian
  if (
    fp.includes('hamiltonian') || 
    name.includes('hamiltonian') || 
    name.includes('spin')
  ) {
    return 'Hamiltonian';
  }

  // 5. Tensoren & Mathematische Strukturen
  if (fp.includes('tensor') || name.includes('tensor')) {
    return 'Tensors';
  }

  // 6. Backends & Execution (Simulatoren, Harware-Schnittstellen)
  if (
    fp.includes('backend') || 
    fp.includes('simulator') || 
    name.includes('backend')
  ) {
    return 'Backends';
  }

  // 7. Performance & Metriken
  if (fp.includes('metric') || name.includes('metric')) {
    return 'Metrics';
  }

  // 8. Permeabilität & Transpilation/Optimization
  if (fp.includes('permeability') || name.includes('permeability')) {
    return 'Permeability';
  }

  // 9. Session-, Speicher- & Session-Management
  if (
    fp.includes('management') || 
    name.includes('management') || 
    fp.includes('session')
  ) {
    return 'QuantumManagement';
  }

  // 10. Kerndatenstrukturen & Variablen (QBool, QInt, QuantumArray, QuantumCircuit etc.)
  // Erst nach den spezifischeren Modulen abfragen, damit nicht alles hier landet
  if (
    fp.includes('core') || 
    fp.includes('circuit') || 
    fp.includes('variable') ||
    name.includes('quantum') || 
    name.includes('qubit') || 
    name.includes('clbit') ||
    name.startsWith('q') // fängt QBool, QInt, QFloat etc. ab
  ) {
    return 'QuantumVariables';
  }

  return 'Other';
};

// Hilfsfunktion für UML-Sichtbarkeiten
const getVisibilitySymbol = (vis) => {
  if (vis === 'public') return '+';
  if (vis === 'protected') return '#';
  if (vis === 'private') return '-';
  return '~';
};

export default function App() {
  const [activeDiagram, setActiveDiagram] = useState('UML');
  const [selectedCategory, setSelectedCategory] = useState('Algorithms & Problems'); // Standard-Kategorie
  
  const paperContainerRef = useRef(null);
  const graphRef = useRef(null);
  const paperRef = useRef(null);

  // 1. JointJS Initialisierung (Graph & Paper) - Läuft nur einmal
  useEffect(() => {
    if (!paperContainerRef.current) return;
    
    if (!graphRef.current) {
      graphRef.current = new joint.dia.Graph({}, { cellNamespace: joint.shapes });
    }

    if (!paperRef.current) {
      paperRef.current = new joint.dia.Paper({
        el: paperContainerRef.current,
        model: graphRef.current,
        width: paperContainerRef.current.clientWidth || 1200,
        height: 800,
        gridSize: 10,
        drawGrid: {
          name: 'dot',
          args: { color: '#e5e7eb', thickness: 1 }
        },
        background: { color: '#ffffff' },
        cellNamespace: joint.shapes,
        interactive: { linkMove: false },
        defaultAnchor: { name: 'center' },
        defaultConnectionPoint: { name: 'boundary' }
      });

      // Simple Panning
      let panning = false;
      let dragStartPosition = null;
      
      paperRef.current.on('blank:pointerdown', function(event) {
        panning = true;
        dragStartPosition = { x: event.clientX, y: event.clientY };
      });

      paperRef.current.on('cell:pointerup blank:pointerup', function() {
        panning = false;
      });

      const container = paperContainerRef.current;
      const handleMouseMove = (event) => {
        if (panning && dragStartPosition && paperRef.current) {
          const currentTranslate = paperRef.current.translate();
          paperRef.current.translate(
            currentTranslate.tx + (event.clientX - dragStartPosition.x),
            currentTranslate.ty + (event.clientY - dragStartPosition.y)
          );
          dragStartPosition = { x: event.clientX, y: event.clientY };
        }
      };

      container.addEventListener('mousemove', handleMouseMove);
    }

    const handleResize = () => {
      if (paperRef.current && paperContainerRef.current) {
        paperRef.current.setDimensions(
          paperContainerRef.current.clientWidth || 1200, 
          800
        );
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 2. Diagramme rendern (wird bei Tab- oder Kategoriewechsel getriggert)
  useEffect(() => {
    if (!graphRef.current || !paperRef.current) return;
    
    const graph = graphRef.current;
    const paper = paperRef.current;

    graph.clear();
    paper.translate(0, 0);
    paper.scale(1, 1);

    if (activeDiagram === 'UML') {
      const umlClassStyle = {
        '.uml-class-name-rect': { fill: '#ffffff', stroke: '#2563eb', strokeWidth: 2 },
        '.uml-class-attrs-rect': { fill: '#eff6ff', stroke: '#2563eb', strokeWidth: 2 },
        '.uml-class-methods-rect': { fill: '#eff6ff', stroke: '#2563eb', strokeWidth: 2 },
        '.uml-class-name-text': { fontFamily: 'sans-serif', fontWeight: '600', fill: '#1e3a8a', fontSize: 14 },
        '.uml-class-attrs-text': { fontFamily: 'sans-serif', fill: '#1e40af', fontSize: 11 },
        '.uml-class-methods-text': { fontFamily: 'sans-serif', fill: '#1e40af', fontSize: 11 }
      };

      // Klassen filtern & gruppieren basierend auf dem Dropdown
      const groupedClasses = {};
      umlData.classes.forEach(cls => {
        const cat = getCategory(cls);
        if (selectedCategory === 'All' || cat === selectedCategory) {
          if (!groupedClasses[cat]) groupedClasses[cat] = [];
          groupedClasses[cat].push(cls);
        }
      });

      const cells = [];
      const classNodeMap = {}; 
      let currentY = 50;

      // Kategorien als Blöcke zeichnen
      Object.keys(groupedClasses).sort().forEach(category => {
        const classes = groupedClasses[category];
        let currentX = 50;
        let rowMaxHeight = 0;

        // Kategorie-Label (Hintergrund-Rechteck dient als Überschrift)
        const categoryLabel = new joint.shapes.standard.Rectangle({
          position: { x: 50, y: currentY },
          size: { width: 350, height: 30 },
          attrs: {
            body: { fill: '#334155', stroke: 'none', rx: 4, ry: 4 },
            label: { text: category.toUpperCase(), fill: '#ffffff', fontWeight: 'bold', fontSize: 14 }
          }
        });
        cells.push(categoryLabel);
        
        currentY += 60; // Platz für Klassen unter der Überschrift

        classes.forEach(cls => {
          const attrs = cls.attributes.map(a => `${getVisibilitySymbol(a.visibility)} ${a.name}: ${a.type}`);
          const meths = cls.methods.map(m => {
            const params = m.parameters.map(p => p.name).join(', ');
            return `${getVisibilitySymbol(m.visibility)} ${m.name}(${params}): ${m.return_type}`;
          });

          // Höhe dynamisch berechnen
          const height = 50 + (attrs.length * 14) + (meths.length * 14);
          rowMaxHeight = Math.max(rowMaxHeight, height);

          const umlClass = new joint.shapes.uml.Class({
            position: { x: currentX, y: currentY },
            size: { width: 320, height: height },
            name: cls.class_name,
            attributes: attrs,
            methods: meths,
            attrs: umlClassStyle
          });

          classNodeMap[cls.class_name] = umlClass;
          cells.push(umlClass);

          currentX += 350; // Horizontaler Abstand nächste Klasse
          
          // Zeilenumbruch
          if (currentX > 1000) {
            currentX = 50;
            currentY += rowMaxHeight + 40;
            rowMaxHeight = 0;
          }
        });

        currentY += rowMaxHeight + 80; // Abstand zur nächsten Kategorie (falls 'All' gewählt ist)
      });

      // Generalisierungs-Links (Vererbung) nur zeichnen, wenn beide Nodes im Filter existieren
      umlData.classes.forEach(cls => {
        const sourceNode = classNodeMap[cls.class_name];
        if (!sourceNode) return;

        cls.parents.forEach(parentName => {
          const targetNode = classNodeMap[parentName];
          if (targetNode) {
            const link = new joint.shapes.uml.Generalization({
              source: { id: sourceNode.id },
              target: { id: targetNode.id },
              router: { name: 'manhattan' },
              connector: { name: 'rounded' }
            });
            cells.push(link);
          }
        });
      });

      graph.addCells(cells);

      // Graph der Größe anpassen, zoomt perfekt auf die gewählte Kategorie
      setTimeout(() => paper.scaleContentToFit({ padding: 60, maxScale: 1.5 }), 100);

    } else if (activeDiagram === 'CENELEC') {
      const createGate = (x, y, label, color) => {
        const gate = new joint.shapes.standard.Rectangle();
        gate.position(x, y);
        gate.resize(100, 60);
        gate.attr({
          body: { fill: color || '#ffffff', stroke: '#334155', strokeWidth: 2, rx: 4, ry: 4 },
          label: { text: label, fill: '#0f172a', fontFamily: 'sans-serif', fontWeight: 'bold', fontSize: 13 }
        });
        return gate;
      };

      const createWire = (source, target) => {
        const link = new joint.shapes.standard.Link();
        link.source({ id: source.id });
        link.target({ id: target.id });
        link.router('orthogonal');
        link.connector('rounded');
        link.attr({
          line: { stroke: '#475569', strokeWidth: 2, targetMarker: { type: 'path', d: 'M 10 -5 0 0 10 5 Z', fill: '#475569' } }
        });
        return link;
      };

      const qb0 = createGate(100, 100, 'q0 (Input)', '#f8fafc');
      const qb1 = createGate(100, 200, 'q1 (Input)', '#f8fafc');
      const qb2 = createGate(100, 300, 'q2 (Input)', '#f8fafc');

      const hGate1 = createGate(300, 100, 'H', '#e0f2fe');
      const cnotGate = createGate(500, 150, 'CNOT', '#bfdbfe');
      cnotGate.resize(100, 110);
      
      const meas1 = createGate(700, 100, 'Measure', '#f1f5f9');
      const meas2 = createGate(700, 200, 'Measure', '#f1f5f9');

      graph.addCells([qb0, qb1, qb2, hGate1, cnotGate, meas1, meas2]);

      const wires = [
        createWire(qb0, hGate1),
        createWire(hGate1, cnotGate),
        createWire(qb1, cnotGate),
        createWire(cnotGate, meas1),
        createWire(cnotGate, meas2)
      ];

      graph.addCells(wires);
      setTimeout(() => paper.scaleContentToFit({ padding: 80, maxScale: 1.2 }), 50);
    }
  }, [activeDiagram, selectedCategory]);

  return (
    <div className="w-full min-h-screen bg-slate-50 flex flex-col font-sans text-slate-800 text-left items-start">
      <header className="w-full bg-white border-b border-slate-200 shadow-sm flex flex-row justify-between items-center px-8 py-4 shrink-0">
        <div className="flex flex-row items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 text-white rounded shadow flex items-center justify-center font-bold text-lg">
            Q
          </div>
          <span className="text-xl font-bold tracking-tight text-slate-900">Qrisp Diagram Viewer</span>
        </div>
        
        <nav className="flex flex-row items-center gap-6 text-sm font-semibold text-slate-600">
          <a href="#" className="hover:text-blue-600 transition-colors">Documentation</a>
          <a href="#" className="hover:text-blue-600 transition-colors">Tutorials</a>
          <a href="#" className="hover:text-blue-600 transition-colors">Framework</a>
        </nav>
      </header>

      <main className="w-full max-w-7xl mx-auto p-6 flex flex-col gap-6 flex-1">
        <div className="grid grid-cols-2 gap-6 w-full shrink-0">
          <div className="bg-white p-8 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-center text-left h-full">
            <h2 className="text-2xl font-bold mb-3 text-slate-900">Introduction</h2>
            <p className="text-slate-600 leading-relaxed text-left">
              Explore the architectural paradigms of the Qrisp framework. Toggle the buttons on the right to switch between the high-level <strong>UML Class Hierarchy</strong> and the compiled <strong>CENELEC Logic Circuits</strong>.
            </p>
          </div>

          <div className="flex flex-col gap-4 h-full w-full">
            <div className="flex flex-row gap-4 flex-1">
              <button 
                onClick={() => setActiveDiagram('UML')}
                className={`flex-1 rounded-xl border-2 flex flex-col items-center justify-center font-bold text-xl transition-all duration-200 focus:outline-none ${
                  activeDiagram === 'UML' 
                    ? 'border-blue-600 bg-blue-50 text-blue-700 shadow-inner' 
                    : 'border-slate-200 bg-white text-slate-600 hover:border-blue-300 hover:bg-slate-50 shadow-sm'
                }`}
              >
                UML
              </button>
              <button 
                onClick={() => setActiveDiagram('CENELEC')}
                className={`flex-1 rounded-xl border-2 flex flex-col items-center justify-center font-bold text-xl transition-all duration-200 focus:outline-none ${
                  activeDiagram === 'CENELEC' 
                    ? 'border-blue-600 bg-blue-50 text-blue-700 shadow-inner' 
                    : 'border-slate-200 bg-white text-slate-600 hover:border-blue-300 hover:bg-slate-50 shadow-sm'
                }`}
              >
                CENELEC
              </button>
            </div>
            
            {/* Filter-Dropdown (Nur sichtbar bei UML) */}
            {activeDiagram === 'UML' && (
              <div className="flex flex-row items-center gap-3 bg-white p-3 rounded-xl border border-slate-200 shadow-sm shrink-0">
                <label htmlFor="category-select" className="text-sm font-bold text-slate-700 whitespace-nowrap">
                  Kategorie filtern:
                </label>
                <select
                  id="category-select"
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="flex-1 bg-slate-50 border border-slate-300 text-slate-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2 outline-none cursor-pointer"
                >
                  <option value="All">Alle (Gesamtübersicht)</option>
                  {CATEGORIES.map(cat => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>

        <div 
          className="w-full border border-[#ccc] bg-white rounded-xl shadow-sm overflow-hidden relative flex flex-col"
          style={{ height: '800px', minHeight: '800px' }}
        >
          <div 
            ref={paperContainerRef} 
            className="w-full h-full cursor-grab active:cursor-grabbing"
            style={{ height: '800px', width: '100%' }}
          ></div>
          
          <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur px-4 py-2 rounded-lg border border-slate-200 shadow-sm text-xs font-semibold text-slate-500 z-10">
            {activeDiagram === 'UML' 
              ? `Showing: UML Diagram (${selectedCategory === 'All' ? 'All Classes' : selectedCategory})` 
              : 'Showing: CENELEC Diagram'}
          </div>
        </div>
      </main>
    </div>
  );
}