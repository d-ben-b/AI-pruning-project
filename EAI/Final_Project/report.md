# 🤖 Efficient AI-Driven Elderly Care Robot
### AI System Lab — Efficient AI Model Design Final Project Proposal  
**Team Name:** TBD  
**Members:** TBD  
**Date:** 2025/11/05  

---

## 🧠 Motivation
隨著台灣進入超高齡社會，長照人力資源逐漸短缺，  
如何運用 AI 技術輔助長者日常照護，成為智慧醫療與社會永續的重要課題。  

本專案旨在開發一款 **低功耗、高效率的長照語音互動與物件追蹤機器人**，  
可於 Jetson Nano 或樹梅派上執行，整合語音辨識、語言理解與即時視覺辨識模組，  
協助長者完成簡單指令、追蹤目標或物品，並能以自然語言互動，達到貼心陪伴與警示功能。  

---

## ⚙️ System Overview
```

🎙️ Voice Input
↓
[Speech-to-Text Model (Distilled Whisper / HuggingFace s2s)]
↓
[Lightweight LLM (Quantized LLaMA / Phi-mini / Mistral-instruct)]
↓
🗣️ Text Response / Voice Output (TTS)
↓
[OpenCV Object Tracking Module]
↓
🎥 Camera → Face / Object Tracking → Motor Control (optional)

```

整個系統將部署於 **Jetson Nano 或 Raspberry Pi 4B**，  
強調 **在無雲端環境下的端側語音理解與互動能力**。

---

## 🧩 Model Design and Efficiency

| 模組 | 模型 | 高效化技術 | 功能 |
|------|------|-------------|------|
| 🎧 **語音辨識 (ASR)** | [HuggingFace `speech-to-speech`](https://github.com/huggingface/speech-to-speech) 或 Distilled Whisper Tiny | Distillation + INT8 Quantization + TensorRT | 將長者語音轉成文字（命令或對話） |
| 💬 **語言理解 (LLM)** | **LLaMA-3-mini / Phi-3-mini (Quantized)** | Quantization (INT8) + 2:4 Structured Pruning | 對話生成與自然語言命令解析 |
| 👀 **視覺模組 (CV)** | OpenCV + YOLOv8-Nano | 模型剪枝 + Depthwise Conv | 追蹤長者 / 識別物件（如杯子、拐杖等） |

---

## 🚀 Deployment Plan

| 項目 | 平台 / 方法 |
|------|---------------|
| **硬體平台** | Jetson Nano / Raspberry Pi 4B |
| **推論框架** | PyTorch → ONNX → TensorRT / TFLite |
| **語音模型優化** | FP16/INT8 TensorRT Engine + Streaming Buffer |
| **LLM 優化** | GPTQ 量化 + Token streaming |
| **視覺模組** | OpenCV DNN 模組 + USB Camera |
| **整合介面** | Python ROS2 node / Flask local server |
| **互動介面** | 語音輸入 + TTS 輸出（可選 PicoTTS） |

---

## 🧮 Efficiency & Evaluation Metrics

| 評估項目 | 指標 | 目標 |
|-----------|--------|-------|
| **延遲 (Latency)** | 語音輸入至回覆時間 | ≤ 1.2 秒 |
| **模型大小** | ASR + LLM + CV 模組總和 | ≤ 400 MB |
| **功耗** | 平均系統功耗 | ≤ 15 W |
| **準確率** | 語音辨識正確率 | ≥ 90% |
| **追蹤穩定度** | IoU 穩定率 | ≥ 85% |

---

## 📅 Timeline

| 週次 | 任務內容 |
|------|-----------|
| **Week 1 (11/5–11/10)** | Whisper / s2s 語音模型整合測試，Jetson Nano 環境建置 |
| **Week 2 (11/11–11/17)** | LLaMA-3-mini 量化與剪枝，建立語音互動 pipeline |
| **Week 3 (11/18–11/24)** | OpenCV 物件偵測與追蹤整合，建立語音控制指令 |
| **Week 4 (11/25–12/01)** | 整合測試、TensorRT 優化、延遲與準確率評估、Demo 與簡報準備 |

---

## 🎯 Expected Results

- 成功於 Jetson Nano / Raspberry Pi 上運行 **離線語音助理**  
- 可識別「開燈 / 追蹤我 / 撿起物品」等命令  
- 實現 **即時語音理解 + 視覺辨識 + 回覆語音輸出**  
- 提供長者語音互動與基本行為輔助，提升生活便利與安全性  

---

## 💡 Innovation & Impact

- **跨模組高效設計**：結合語音、語言與視覺三大 AI 模組於低功耗硬體上運作。  
- **人性化應用導向**：針對長照情境設計互動介面與安全機制。  
- **技術深度**：融合 Distillation、Quantization、Pruning 與 Edge Deployment。  
- **社會價值**：對應台灣高齡化社會需求，具實際推廣潛力。  

---

## 📚 Reference
- HuggingFace Speech-to-Speech: [https://github.com/huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)  
- OpenAI Whisper & Distil-Whisper  
- LLaMA-3 / Phi-3-mini Quantized models (HuggingFace)  
- NVIDIA TensorRT Developer Guide  
- OpenCV Object Tracking (CSRT / MOSSE / DeepSORT)

---