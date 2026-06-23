# ckd_app/utils/seed_rag.py
import os
import sys

# Ensure parent directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ckd_app.utils.rag_engine import CKDRAGEngine

def seed_database():
    guidelines_path = "data/kdigo_guidelines.txt"
    db_path = "data/clinical_knowledge_base.json"
    
    print("🚀 Initializing CKD Clinical Knowledge Base Seeding...")
    
    if not os.path.exists(guidelines_path):
        print(f"❌ Error: Guidelines source file not found at '{guidelines_path}'")
        return False
        
    try:
        with open(guidelines_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Split by section marker "---" or segment blocks
        sections = content.split("---")
        
        # Initialize RAG engine
        engine = CKDRAGEngine(db_path=db_path)
        
        # Clear existing documents to avoid double-seeding
        engine.documents = []
        engine.embeddings = []
        
        for idx, section in enumerate(sections):
            section_text = section.strip()
            if not section_text:
                continue
                
            # Extract first line as a title or generate a default one
            lines = [l.strip() for l in section_text.split("\n") if l.strip()]
            if lines:
                title = lines[0].replace(":", "").replace("#", "").strip()
            else:
                title = f"KDIGO Guideline Section {idx + 1}"
                
            print(f"📦 Indexing section: '{title}' ({len(section_text)} chars)...")
            engine.add_document(title=title, text=section_text, category="guidelines")
            
        print("✅ RAG Database successfully seeded and saved!")
        return True
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    seed_database()
