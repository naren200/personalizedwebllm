import React from 'react';
import ElevatorPitchChat from './components/ElevatorPitchChat';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>🚀 Elevator Pitch Bot</h1>
        <p>React Integration Demo</p>
      </header>
      <main className="App-main">
        <ElevatorPitchChat />
      </main>
      <footer className="App-footer">
        <p>Powered by WebLLM and React</p>
      </footer>
    </div>
  );
}

export default App;