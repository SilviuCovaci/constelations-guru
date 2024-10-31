# Proiect Demo: Detectare și Clasificare Constelații din Imagini

Acest proiect demonstrează utilizarea unor modele de învățare profundă pentru detectarea și clasificarea constelațiilor, precum și un sistem expert pentru identificarea lor. Modelele implementate includ YOLO și Faster R-CNN pentru detectare, respectiv RNN și Swin Transformers pentru clasificare.

## Tehnologii și Biblioteci
- Python
- FastAPI pentru server și endpoint-uri
- Uvicorn pentru server ASGI
- Jinja2 pentru template-uri
- OpenCV, PyTorch, YOLO, și alte librării de ML pentru modele

## Instalare

1. **Clonați proiectul**:
   ```bash
   git clone https://github.com/SilviuCovaci/constelations-guru
   cd constelations-guru

2. **Instalați dependențele**:
Asigurați-vă că aveți un mediu Python configurat (recomandat versiunea 3.8+).

3. ** Copiere Modele **
Descărcați modelele folosite in demo și dezarhivați-le în directorul app/models

https://drive.google.com/file/d/1zX9-593VsI902AAFRGSJBGo4ACOBgUvY/view?usp=sharing


## Rulare
Pentru a porni serverul local, folosiți comanda:
uvicorn app.main:app --reload --port=9595
