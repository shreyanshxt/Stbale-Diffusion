import React, { useState, useRef } from 'react';
import { Upload, Wand2, Download, Settings, X, Loader2 } from 'lucide-react';

const StableDiffusionUI = () => {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [inputImage, setInputImage] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [mode, setMode] = useState('text2img');
  
  // Settings
  const [settings, setSettings] = useState({
    cfgScale: 8,
    strength: 0.9,
    steps: 50,
    seed: 42,
    width: 512,
    height: 512
  });

  const fileInputRef = useRef(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setInputImage(e.target.result);
        setMode('img2img');
      };
      reader.readAsDataURL(file);
    }
  };

  const generateImage = async () => {
    setLoading(true);
    setOutputImage(null);

    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('negative_prompt', negativePrompt);
      formData.append('cfg_scale', settings.cfgScale);
      formData.append('steps', settings.steps);
      formData.append('seed', settings.seed);
      formData.append('width', settings.width);
      formData.append('height', settings.height);
      
      if (mode === 'img2img' && inputImage) {
        formData.append('input_image', inputImage);
        formData.append('strength', settings.strength);
      }

      const response = await fetch('http://localhost:5000/generate', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Generation failed');

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setOutputImage(imageUrl);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to generate image. Make sure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = () => {
    if (!outputImage) return;
    const a = document.createElement('a');
    a.href = outputImage;
    a.download = `sd-generation-${Date.now()}.png`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-3 bg-gradient-to-r from-pink-400 to-purple-400 bg-clip-text text-transparent">
            Stable Diffusion Studio
          </h1>
          <p className="text-gray-300">Transform your imagination into reality</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel - Controls */}
          <div className="space-y-6">
            {/* Mode Selection */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <div className="flex gap-4 mb-6">
                <button
                  onClick={() => setMode('text2img')}
                  className={`flex-1 py-3 rounded-xl font-semibold transition-all ${
                    mode === 'text2img'
                      ? 'bg-gradient-to-r from-pink-500 to-purple-500 shadow-lg'
                      : 'bg-white/5 hover:bg-white/10'
                  }`}
                >
                  Text to Image
                </button>
                <button
                  onClick={() => setMode('img2img')}
                  className={`flex-1 py-3 rounded-xl font-semibold transition-all ${
                    mode === 'img2img'
                      ? 'bg-gradient-to-r from-pink-500 to-purple-500 shadow-lg'
                      : 'bg-white/5 hover:bg-white/10'
                  }`}
                >
                  Image to Image
                </button>
              </div>

              {/* Prompt Input */}
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Prompt</label>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe your image... (e.g., A majestic dragon flying over mountains at sunset)"
                    className="w-full bg-white/5 border border-white/20 rounded-xl p-4 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                    rows="4"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Negative Prompt</label>
                  <textarea
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="What to avoid... (e.g., blurry, low quality, distorted)"
                    className="w-full bg-white/5 border border-white/20 rounded-xl p-4 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                    rows="2"
                  />
                </div>
              </div>

              {/* Image Upload for img2img */}
              {mode === 'img2img' && (
                <div className="mt-4">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleImageUpload}
                    accept="image/*"
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full py-4 border-2 border-dashed border-white/30 rounded-xl hover:border-purple-400 transition-all flex items-center justify-center gap-2"
                  >
                    <Upload size={20} />
                    {inputImage ? 'Change Input Image' : 'Upload Input Image'}
                  </button>
                  {inputImage && (
                    <div className="mt-4 relative">
                      <img src={inputImage} alt="Input" className="w-full rounded-xl" />
                      <button
                        onClick={() => setInputImage(null)}
                        className="absolute top-2 right-2 bg-red-500 p-2 rounded-full hover:bg-red-600"
                      >
                        <X size={16} />
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Settings */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="flex items-center justify-between w-full mb-4"
              >
                <span className="font-semibold flex items-center gap-2">
                  <Settings size={20} />
                  Advanced Settings
                </span>
                <span className="text-2xl">{showSettings ? '−' : '+'}</span>
              </button>

              {showSettings && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm mb-2">CFG Scale: {settings.cfgScale}</label>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      step="0.5"
                      value={settings.cfgScale}
                      onChange={(e) => setSettings({...settings, cfgScale: parseFloat(e.target.value)})}
                      className="w-full"
                    />
                  </div>

                  {mode === 'img2img' && (
                    <div>
                      <label className="block text-sm mb-2">Strength: {settings.strength}</label>
                      <input
                        type="range"
                        min="0.1"
                        max="1"
                        step="0.1"
                        value={settings.strength}
                        onChange={(e) => setSettings({...settings, strength: parseFloat(e.target.value)})}
                        className="w-full"
                      />
                    </div>
                  )}

                  <div>
                    <label className="block text-sm mb-2">Steps: {settings.steps}</label>
                    <input
                      type="range"
                      min="20"
                      max="100"
                      step="10"
                      value={settings.steps}
                      onChange={(e) => setSettings({...settings, steps: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm mb-2">Seed</label>
                    <input
                      type="number"
                      value={settings.seed}
                      onChange={(e) => setSettings({...settings, seed: parseInt(e.target.value)})}
                      className="w-full bg-white/5 border border-white/20 rounded-xl p-3 text-white"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Generate Button */}
            <button
              onClick={generateImage}
              disabled={loading || !prompt}
              className="w-full py-4 bg-gradient-to-r from-pink-500 to-purple-500 rounded-xl font-bold text-lg hover:shadow-2xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" size={24} />
                  Generating...
                </>
              ) : (
                <>
                  <Wand2 size={24} />
                  Generate Image
                </>
              )}
            </button>
          </div>

          {/* Right Panel - Output */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold">Generated Image</h2>
              {outputImage && (
                <button
                  onClick={downloadImage}
                  className="flex items-center gap-2 bg-green-500 hover:bg-green-600 px-4 py-2 rounded-lg transition-all"
                >
                  <Download size={20} />
                  Download
                </button>
              )}
            </div>

            <div className="aspect-square bg-white/5 rounded-xl flex items-center justify-center overflow-hidden border-2 border-white/20">
              {loading ? (
                <div className="text-center">
                  <Loader2 className="animate-spin mx-auto mb-4" size={48} />
                  <p className="text-gray-300">Creating your masterpiece...</p>
                </div>
              ) : outputImage ? (
                <img src={outputImage} alt="Generated" className="w-full h-full object-contain" />
              ) : (
                <div className="text-center text-gray-400">
                  <Wand2 size={48} className="mx-auto mb-4 opacity-50" />
                  <p>Your generated image will appear here</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StableDiffusionUI;