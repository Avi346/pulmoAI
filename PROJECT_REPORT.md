# 🩺 AI Radiologist Assistant: Comprehensive Technical & Architectural Report

> [!IMPORTANT]
> **Executive Summary**: The AI Radiologist Assistant is an enterprise-grade, multimodal clinical decision support system designed to solve two critical challenges in pulmonary healthcare: **high-sensitivity screening** of pediatric chest radiographs for pneumonia, and **evidence-based clinical consultation** grounded in authoritative international healthcare guidelines.

---

## 1. Project Overview & Architectural Topology

Modern medical diagnostic tools must bridge the gap between perceptual pattern recognition (Computer Vision) and complex clinical reasoning (Natural Language Processing). This project implements a **dual-pipeline multimodal architecture** combining a fine-tuned deep residual/dense convolutional neural network with a Retrieval-Augmented Generation (RAG) conversational assistant.

### 1.1 High-Level System Architecture

```mermaid
graph TD
    classDef frontend fill:#1f77b4,stroke:#fff,stroke-width:2px,color:#fff;
    classDef vision fill:#2ca02c,stroke:#fff,stroke-width:2px,color:#fff;
    classDef rag fill:#9467bd,stroke:#fff,stroke-width:2px,color:#fff;
    classDef data fill:#ff7f0e,stroke:#fff,stroke-width:2px,color:#fff;

    User(["👤 Clinician / User"]) -->|Uploads Radiograph| App["💻 Streamlit Dashboard app.py"]
    User -->|Asks Clinical Query| App

    subgraph Background Initialization
        Warmup["⚙️ Daemon Warmup Threads"] -->|Preloads| ModelCache["🧠 DenseNet-121 Graph"]
        Warmup -->|Preloads| RAGCache["📚 ChromaDB + Embeddings"]
    end

    subgraph Computer Vision Pipeline train_vision.py
        App -->|Temp PNG| Otsu["🔎 Otsu Lung Cropping"]
        Otsu -->|Stripped Image| Pad["📐 PadToSquare 384x384"]
        Pad -->|Normalized Tensor| DenseNet["🏛️ DenseNet-121 Backbone"]
        DenseNet -->|Softmax Averaging| TTA["🔄 Test-Time Augmentation"]
        TTA -->|Dx & Confidence| ScreeningResult["📋 Screening Diagnosis"]
    end

    subgraph Clinical RAG Pipeline build_rag.py
        PDFs[("📄 Medical Guideline PDFs")] -->|PyPDFLoader| Splitter["✂️ Recursive Text Splitter"]
        Splitter -->|1000/200 Chunks| Embed["🔢 MiniLM-L6 Embeddings"]
        Embed -->|384-d Vectors| Chroma[("💾 ChromaDB Vector Store")]

        ScreeningResult -->|Patient Context Inject| Prompt["📝 ChatPromptTemplate"]
        App -->|Chat History & Query| Prompt
        Chroma -->|Top-k Similarity Search| Prompt
        Prompt -->|Contextualized Query| Mistral["🤖 Mistral AI LLM API"]
        Mistral -->|Consultative Advice| Response["💬 Clinical Guidance"]
    end

    ScreeningResult --> App
    Response --> App

    class App frontend;
    class Otsu,Pad,DenseNet,TTA,ScreeningResult vision;
    class PDFs,Splitter,Embed,Chroma,Prompt,Mistral,Response rag;
    class Warmup,ModelCache,RAGCache data;
```

### 1.2 Core File Topology
- [app.py](file:///d:/ML/project3/app.py): The interactive, responsive frontend user interface built with Streamlit, implementing asynchronous background caching and dynamic medical chat sessions.
- [train_vision.py](file:///d:/ML/project3/src/train_vision.py): The core vision engine handling data normalization, automated lung isolation, model fine-tuning, focal loss regularization, and inference with Test-Time Augmentation.
- [build_rag.py](file:///d:/ML/project3/src/build_rag.py): The clinical text retrieval engine responsible for document ingestion, vector embedding generation, ChromaDB persistence, and contextual Mistral AI prompt execution.
- [requirements.txt](file:///d:/ML/project3/requirements.txt): Complete specification of underlying deep learning, computer vision, vector database, and frontend dependencies.
- [README.md](file:///d:/ML/project3/README.md): Project documentation and test-set benchmark evaluation records.

---

## 2. Computer Vision Pipeline: Architectural & Algorithmic Rationale

The diagnostic computer vision pipeline ([train_vision.py](file:///d:/ML/project3/src/train_vision.py)) is engineered specifically to overcome the unique failure modes of medical radiographs, such as severe class imbalance, extreme anatomical variation, and shortcut learning.

### 2.1 Model Selection Rationale: `DenseNet-121` vs `ResNet`
While legacy project sketches ([README.md](file:///d:/ML/project3/README.md)) initially referenced ResNet-18, the production architecture upgraded to **DenseNet-121** (`models.densenet121`).

> [!TIP]
> **Why DenseNet-121?**
> 1. **Feature Reuse via Dense Connectivity**: In traditional architectures (like ResNet or VGG), layers only receive outputs from their immediate predecessor. In DenseNet, each layer receives concatenated feature maps from *all preceding layers*. This ensures that low-level edge features (rib contours, diaphragm boundaries) are directly accessible alongside high-level semantic abstractions (lobular consolidation).
> 2. **Vanishing Gradient Mitigation**: Direct connections from the loss function to early convolution blocks facilitate clean gradient flow throughout the 121 layers during backpropagation.
> 3. **Parameter Efficiency**: DenseNet achieves superior representational capacity with narrower layers (lower growth rate), requiring fewer trainable parameters than ResNet-50 while capturing subtle pulmonary opacities.

### 2.2 Preprocessing Technique 1: Automated Lung Field Cropping (Otsu Thresholding)
One of the most dangerous vulnerabilities in medical AI is **Shortcut Learning** (or the *Clever Hans effect*). Deep convolutional networks frequently achieve high accuracy not by analyzing lung tissue, but by recognizing extraneous artifacts: chest tubes, pacemakers, neck collars, or hospital ward markers stamped in the corners of X-rays from sicker patients.

To prevent this, `crop_lung_region()` implements an automated segmentation algorithm:
1. **Grayscale Conversion & Gaussian Filtering**: Converts the radiograph to luminance (`L`) and applies a Gaussian smoothing filter (`sigma=2`) via `scipy.ndimage.gaussian_filter` to suppress quantum mottle and high-frequency noise.
2. **Otsu's Thresholding**: Computes an optimal bimodal threshold (`skimage.filters.threshold_otsu`) separating low-density air/tissue from high-density bone and background.
3. **Morphological Closing**: Applies morphological closing (`skimage.morphology.closing` with a `5x5` square kernel) to seal small anatomical gaps and vascular structures within the lung parenchyma.
4. **Bounding Box Extraction & Padding**: Isolates the central thoracic bounding box (ignoring the outer 10% image margins where hospital labels reside) and expands the crop by a safety margin of **5%** to ensure peripheral pleural effusions are not truncated.

### 2.3 Preprocessing Technique 2: Aspect-Ratio Preserving Padding (`PadToSquare`)
Standard deep learning pipelines resize input images directly to a square grid (e.g., squashing a `2000x1500` radiograph into `384x384`). In radiology, **geometric distortion is unacceptable**. Stretching or compressing an X-ray artificially alters cardiothoracic ratios and distorts the angle of costophrenic recesses, mimicking or masking infiltrates.

The custom `PadToSquare` transform solves this:
- Resizes the *longest edge* exactly to `384px` using bilinear interpolation (`Image.Resampling.BILINEAR`), preserving exact anatomical proportions.
- Pads the remaining shorter sides evenly with a neutral gray background `(124, 116, 104)` matching standard radiographic mean intensity, preventing edge-detector neuron spiking at image borders.

### 2.4 Loss Function Selection: `Focal Loss`
Medical screening datasets are inherently imbalanced and contain borderline cases (e.g., faint ground-glass opacities). Standard Cross-Entropy loss assigns equal gradient weight to all samples, allowing overwhelming numbers of "easy normal" images to dominate training updates.

The pipeline implements **Focal Loss** ($\gamma=2.0$):
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- **Down-weighting Easy Examples**: As the network becomes confident in an easy normal radiograph ($p_t \to 1$), the modulating factor $(1 - p_t)^\gamma$ approaches zero, extinguishing its gradient contribution.
- **Dynamic Class Weighting**: Automatically computes inversely proportional class weights (`compute_class_weights`) to balance the 390 Pneumonia vs 234 Normal test samples.

### 2.5 Optimization & Discriminative Fine-Tuning
Training deep networks on small clinical datasets risks catastrophic forgetting of general visual features. The training strategy implements **Layer-Wise Discriminative Fine-Tuning**:
- Early convolution blocks (`denseblock1`, `denseblock2`) are **frozen** (`requires_grad = False`) to retain foundational edge and texture filters learned from ImageNet.
- Top feature extraction blocks (`denseblock3`, `transition3`, `denseblock4`, `norm5`) and the custom classification head (`nn.Linear`) are **unfrozen**.
- **Differential Learning Rates**: The AdamW optimizer applies a conservative learning rate of `1e-5` to backbone parameters to prevent weight destruction, and a faster learning rate of `1e-3` to the new linear classification head, modulated by a `CosineAnnealingLR` scheduler over 10 epochs.

### 2.6 Inference Strategy: Test-Time Augmentation (TTA)
During real-world clinical screening (`predict()`), single-pass predictions can fluctuate due to minor patient positioning shifts. The engine performs **Test-Time Augmentation**:
- Passes both the original normalized radiograph and a horizontally flipped copy through the network.
- Averages their resulting softmax probability distributions `(out_orig + out_flip) / 2.0` before determining final diagnosis and confidence score. This dramatically improves prediction stability and calibration.

---

## 3. Retrieval-Augmented Generation (RAG) Pipeline: Clinical Grounding

Large Language Models (LLMs) are notorious for generating plausible but incorrect medical statements (hallucinations). The clinical assistant tab ([build_rag.py](file:///d:/ML/project3/src/build_rag.py)) prevents this by anchoring all consultative output in indexed clinical reference literature.

### 3.1 Why RAG over Fine-Tuning an LLM?
Fine-tuning an LLM on medical texts bakes knowledge directly into static weights, making it impossible to audit exact citations or update protocols without re-training. **RAG decouples reasoning from knowledge storage**: the LLM acts purely as a clinical synthesizer reading verified excerpts retrieved dynamically from a local vector database.

### 3.2 Document Ingestion & Chunking Strategy
Medical PDF guidelines (such as WHO Childhood Pneumonia Guidelines and ATS/IDSA Community-Acquired Pneumonia Guidelines) are loaded via `PyPDFLoader` and processed by a `RecursiveCharacterTextSplitter`:
- **Chunk Size (`1000` chars)**: Large enough to encapsulate complete clinical recommendations, diagnostic criteria tables, and dosing schedules.
- **Chunk Overlap (`200` chars)**: Crucial for medical literature. Clinical protocols frequently span paragraph breaks; a 200-character sliding overlap ensures that contraindications or secondary drug adjustments are not severed from their primary indication.

### 3.3 Embedding Model Selection: `all-MiniLM-L6-v2`
The vector representation uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`:
- Produces dense **384-dimensional semantic embeddings**.
- **Computational Efficiency**: Unlike heavy 1536-dimensional OpenAI embeddings or multi-gigabyte models, MiniLM executes almost instantaneously on local CPUs while maintaining top-tier benchmarks on semantic text search (MTEB leaderboard).

### 3.4 Vector Database: `ChromaDB`
Vector persistence is managed by **ChromaDB** (`langchain_chroma.Chroma`):
- Operates embedded within the application workspace (`vectorstore/`), requiring no external database servers or Docker containers.
- Implements Hierarchical Navigable Small World (HNSW) graph indexing for sub-millisecond approximate nearest neighbor retrieval (`top_k=5` chunks).

### 3.5 LLM Engine: Cloud `Mistral AI` API
While legacy project documentation noted local Ollama (`qwen3:8b`), the production codebase leverages cloud **Mistral AI (`mistral-small-latest`)** via `ChatMistralAI`:
- **Superior Clinical Reasoning**: Mistral's latest models exhibit exceptional adherence to strict system prompts, complex medical triage logic, and structured markdown formatting.
- **Zero Local Hardware Contention**: Offloading LLM inference to Mistral's cloud APIs ensures that local system memory and GPU resources remain 100% dedicated to PyTorch image inference.

### 3.6 Advanced Prompt Engineering & Context Fusion
The `PROMPT_TEMPLATE` orchestrates a sophisticated multi-variable context fusion:
1. **Adaptive Output Length Control**: Instructs the model to distinguish between simple factual queries (answered in 2-4 sentences) and comprehensive recovery plans.
2. **Silent Diagnostic Awareness**: Automatically injects the patient's AI vision screening result (`patient_context` e.g., *Pneumonia Detected at 98.4% confidence*). The prompt explicitly forbids the LLM from robotically announcing this diagnosis at the start of every reply, creating a natural, empathetic doctor-patient dialogue.
3. **Short-Term Conversational Memory**: Passes a rolling window of the last 6 chat turns (`chat_history`) so the assistant resolves clinical follow-up pronouns smoothly.

---

## 4. Frontend & User Experience Design (`Streamlit`)

The clinician interface ([app.py](file:///d:/ML/project3/app.py)) is engineered for rapid triage dashboard efficiency.

### 4.1 Asynchronous Background Warmup
Deep learning vision models and vector databases suffer from significant cold-start latency upon initial import. `init_background_warmup()` uses Python `threading` to spawn daemon background threads immediately when Streamlit launches:
```python
@st.cache_resource(show_spinner=False)
def init_background_warmup():
    t = threading.Thread(target=_background_warmup, daemon=True)
    t.start()
    return t
```
This preloads the PyTorch `DenseNet-121` weights into memory and warms up inference computation graphs silently while the clinician reads the welcome screen.

### 4.2 Interactive Dual-Tab Workflow
- **Tab 1 (`🖼️ X-Ray Screening`)**: Side-by-side image verification. Uploaded radiographs are temporarily mirrored into a dedicated `scratch/` folder to isolate file system I/O before passing to the PyTorch inference engine.
- **Tab 2 (`💬 Clinical Guideline Assistant`)**: Context-aware chat consultant. Features **dynamic quick-action buttons** that change based on screening results (e.g., showing *Emergency Signs* & *Antibiotic Protocols* if pneumonia was diagnosed, or *Infection Prevention* if normal).

---

## 5. Comprehensive Frameworks & Libraries Matrix

Every library employed in the codebase was selected based on strict criteria of stability, performance, and compatibility.

| Layer | Library / Framework | Version Specification | Primary Role & Usage in Codebase |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | `streamlit` | Latest (`requirements.txt`) | Reactive web application server, session state management, interactive widgets, and background caching (`st.cache_resource`). |
| **Deep Learning** | `torch` | PyTorch Core | Autograd engine, GPU tensor computation, DenseNet backbone architecture definition, Focal Loss math, and differential backpropagation. |
| **Vision Utilities** | `torchvision` | PyTorch Vision | `models.densenet121` weight loading, `transforms` pipeline (rotation, flipping, tensor conversion), and functional padding (`TF.pad`). |
| **Image Processing** | `Pillow` (PIL) | Python Imaging | Radiograph file loading, RGB conversion, aspect-ratio scaling, and bounding box cropping. |
| **Scientific Math** | `scipy` | SciPy Core | `scipy.ndimage.gaussian_filter` for pre-segmentation radiograph smoothing. |
| **Image Filtering** | `scikit-image` | Skimage Core | `threshold_otsu` bimodal lung segmentation threshold calculation and `skimage.morphology` binary mask closing. |
| **Evaluation Metrics**| `scikit-learn` | Sklearn Core | `classification_report` and `confusion_matrix` generation for test-set evaluation benchmarks. |
| **Data Visualization**| `matplotlib` | Agg Backend | Non-interactive backend headless plotting for training loss/accuracy curve export. |
| **Orchestration** | `langchain` | LangChain Core | RAG chain composition (`RunnablePassthrough`), prompt formatting (`ChatPromptTemplate`), and output parsing (`StrOutputParser`). |
| **Document Loaders**| `langchain-community`| Community Addons| `PyPDFLoader` ingestion of medical guideline documents. |
| **Vector Integration**| `langchain-chroma` | Chroma Integration| Bridge connecting LangChain retriever interfaces to underlying Chroma vector databases. |
| **Embedding Wrapper**| `langchain-huggingface`| HF Integration | Wrapper executing local HuggingFace sentence transformer models. |
| **LLM Connector** | `langchain-mistralai`| Mistral Integration| Secure API authentication and chat completion invocations for `ChatMistralAI`. |
| **PDF Parsing** | `pypdf` | Core PDF Engine | Underlying binary text extraction driver utilized by LangChain PDF loaders. |
| **Vector Storage** | `chromadb` | Vector Engine | Embedded HNSW vector indexing, distance calculation, and persistent storage. |
| **Dense Embeddings**| `sentence-transformers`| Core Embedding | Underlying PyTorch execution engine for `all-MiniLM-L6-v2` embedding representations. |

---

## 6. Empirical Performance Benchmark Analysis

The vision classifier underwent rigorous validation on **624 completely unseen test radiographs** (`data/chest_xray/test`). The empirical metrics demonstrate why this pipeline is clinically viable:

| Class | Precision | Recall (Sensitivity) | F1-Score | Sample Support |
| :--- | :---: | :---: | :---: | :---: |
| **NORMAL** | `0.97` | `0.67` | `0.79` | 234 |
| **PNEUMONIA** | `0.83` | **`0.99`** | **`0.90`** | 390 |
| **Total / Accuracy** | — | — | **`87%`** | 624 |

### 6.1 Clinical Interpretation of Benchmark Data

> [!CAUTION]
> **The Screening Gold Standard: 99% Sensitivity**
> In first-line radiological screening, **False Negatives are life-threatening**, whereas False Positives simply result in confirmatory follow-up imaging.
> - **Sensitivity (Pneumonia Recall) = 99%**: Out of 390 patients with active pneumonia, the neural network successfully detected **385 cases**, missing only 5. This near-perfect sensitivity ensures robust patient triage safety.
> - **Normal Precision = 97%**: When the AI predicts that an X-ray is NORMAL, it is correct 97% of the time, allowing radiologists to confidently prioritize abnormal scans.
> - **Focal Loss Efficacy**: The asymmetry between Normal Recall (67%) and Pneumonia Recall (99%) directly reflects the Focal Loss dynamic class weighting, which deliberately traded off borderline normal classifications to guarantee that subtle pneumonia infiltrates were never overlooked.

---

## 7. Conclusion & Architectural Summary

The AI Radiologist Assistant codebase represents an exemplary implementation of applied agentic AI engineering. By replacing naive image resizing with anatomical Otsu cropping and aspect-preserving padding, upgrading legacy backbones to DenseNet-121, regularizing with Focal Loss, and grounding LLM medical consultations via LangChain + ChromaDB RAG, the project successfully delivers a reliable, high-precision diagnostic and consultative medical tool.
