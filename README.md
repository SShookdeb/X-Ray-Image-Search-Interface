# 🩻 X-Ray Image Search Interface  
*A Hybrid Image and Text Retrieval System for Medical X-Ray Data*

---

## 📌 Project Overview

The **X-Ray Image Search Interface** is a hybrid medical image retrieval system designed
to search and retrieve X-ray images using **both textual metadata and visual similarity**.

The project is built on a **modified X-ray dataset** enriched with **500 custom metadata records**,
allowing users to perform **text-based searches** using descriptive attributes as well as
**image-based searches** using deep learning–based visual embeddings.

This dual-search approach closely reflects real-world medical image archive systems used
in hospitals, research labs, and diagnostic centers.

---

## 🎯 Project Objectives

- To create and manage **custom metadata** for X-ray images  
- To modify and structure an existing dataset for efficient retrieval  
- To design a **text-based search engine** using metadata  
- To implement an **image-based similarity search engine** using deep learning  
- To integrate both search approaches into a single interface  

---

## 📊 Dataset & Metadata Design

### 🔹 Modified Dataset
- The original X-ray dataset was **cleaned, restructured, and reorganized**
- Images were renamed and indexed consistently
- The dataset was optimized for embedding generation and fast retrieval

### 🔹 Custom Metadata (500 Records)
- A metadata file containing **500 custom entries** was created
- Each entry is mapped to an individual X-ray image
- Metadata enables **keyword-based and attribute-based search**

📌 *The dataset and embeddings are excluded from this repository to maintain best practices.*

---

## 🔍 Search Engines Implemented

### 1️⃣ Text-Based Search Engine

The **text-based search engine** retrieves X-ray images based on
**metadata attributes rather than image content**.

**Key Characteristics:**
- Operates on the custom metadata file
- Supports keyword and attribute-based queries
- Enables fast and interpretable search
- Useful when visual similarity alone is insufficient

📁 Implementation:
```

src/text_search.py

---

### 2️⃣ Image-Based Search Engine

The **image-based search engine** retrieves X-ray images based on
**visual similarity** using deep learning embeddings.

**How it Works:**
- A pretrained **ResNet** model is used as a feature extractor
- The final classification layer is removed
- Each image is represented as a numerical feature embedding
- Embeddings are **L2-normalized**
- **Cosine similarity** is used to rank images

📁 Implementation:
```

src/image_search.py

````

---

## 🧠 Core Image Search Logic

```python
similarities = self.embeddings @ query_emb
top_indices = np.argsort(similarities)[::-1][:top_k]
````

---

## 🧩 System Architecture

```
User Query
   ├── Text Query  ──► Text Search Engine  ──► Metadata-Based Results
   └── Image Query ──► Image Search Engine ──► Visually Similar Images
```

---

## 📁 Project Structure

```
X-Ray-Image-Search-Interface/
├── src/
│   ├── image_search.py      # Image-based similarity search
│   ├── text_search.py       # Metadata-based text search
│   ├── preprocess.py        # Dataset preprocessing & embedding generation
│   └── utils.py             # Model loading and image transforms
├── metadata/                # Custom metadata (500 records)
├── app.py                   # Application entry point
├── requirements.txt         # Project dependencies
└── .gitignore               # Excludes dataset, embeddings, and environment files
```

---

## ▶️ How to Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

📌 *Dataset and embeddings must be generated separately and are not included in the repository.*

---

## 🔒 Design Considerations

* Text and image search engines are **modular and independent**
* Identical preprocessing is used during indexing and querying
* Embeddings are normalized to ensure correct similarity computation
* Model runs in `eval()` mode for stable feature extraction
* Dataset and embeddings are excluded from version control

---

## 🎥 Prototype Demonstration

🔗 **[https://youtu.be/0G3zfP7lalo](https://youtu.be/0G3zfP7lalo)**

The video demonstrates:

* Text-based search using custom metadata
* Image-based search using visual similarity
* End-to-end system workflow

---

## 🔮 Future Enhancements

* Multi-modal search combining text and image queries
* FAISS-based large-scale similarity search
* CLIP-based joint image–text embeddings
* Web or Streamlit-based user interface
* Support for additional medical imaging modalities

---

## 🎓 Applications

* Medical image retrieval systems
* Hospital and clinical image archives
* Computer vision and deep learning coursework
* Academic mini-projects and capstone projects
* Research and experimental prototypes

---

## ⭐ Conclusion

The **X-Ray Image Search Interface** demonstrates a practical and extensible solution for
medical image retrieval by combining a **modified dataset**, **500 custom metadata entries**,
and both **text-based** and **image-based search engines**.

---

⭐ *If you find this project useful, consider starring the repository.*
