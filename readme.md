**Assistant_d'Analyse_de_Documents_2.1-2.3.ipynb** completes 2.1-2.3, avec un plateforme colab

**Assistant_d'Analyse_de_Documents.ipynb** completes 2.1-2.3, experimental environment python 3.11 (tools vs code) [spécifie explicitement les tokens, impose l'utilisation de SentencePiece et évite les erreurs de conversion tiktoken].

2.4 Interface utilisateur

Développer une interface web à l'aide d'un framework tel que Streamlit et permettez aux utilisateurs de télécharger des fichiers, de poser des questions et d'interagir avec l'assistant. Voici l'implémentation dans app.py. Après la précédente transformation du code, answer_question utilisait Camembert pour les questions-réponses, mais les phrases de réponse étaient trop courtes et pas assez précises. J'ai donc utilisé le modèle mistral de Hugging Face, augmenté la longueur des réponses et essayé plusieurs séries de réponses.

![img](./ressources/plateform_res.jpg)

Assistant d'Analyse de Documents

3 Extensions optionnelles
**- Support multilingue : permettre l’extraction de texte et la réponse aux questions dans plusieurs langues.**

J'utilise la bibliothèque langdetect pour détecter la langue des documents téléchargés. En fonction de la langue détectée (français ou anglais), le modèle correspondant sera chargé (etalab-ia/camembert-base-squadFR-fquad-piaf pour le français, distilbert-base-uncased-distilled-squad pour l'anglais).
Cela permet à l'utilisateur d'extraire du texte et de poser des questions dans plusieurs langues. Si le document est en français, le modèle français est utilisé ; s'il est en anglais, le modèle anglais est utilisé.
![img](./ressources/Support%20multilingue.jpg)

**- Intégration vocale : permettre aux utilisateurs d’interagir avec l’assistant via des commandes vocales.**

La fonction record_audio() permet la saisie vocale. Lorsque l'utilisateur clique sur le bouton « 🎤 Poser une question par voix », le système écoute la question de l'utilisateur à l'aide du microphone et la traite à l'aide de l'API de conversion de la parole en texte de Google (recognize_google), ce qui permet une interaction vocale avec l'assistant.

Les utilisateurs disposent ainsi d'un moyen intuitif d'interagir avec l'assistant sans avoir à taper au clavier.

![img](./ressources/Intégration%20vocale.jpg)

**- Mécanisme de feedback : implémenter un système de retour utilisateur pour affiner les réponses de l’assistant au fil du temps.**

Après avoir fourni une réponse, les utilisateurs sont invités à réagir en demandant s'ils ont trouvé la réponse utile (« Avez-vous trouvé la réponse utile ? »).

S'il sélectionne « Non », il peut fournir des commentaires supplémentaires sur la manière dont la réponse peut être améliorée. Ces commentaires sont stockés dans un fichier texte (feedback.txt), ce qui permet d'affiner les réponses au fil du temps.

![img](./ressources/Mécanisme%20de%20feedback1.jpg)
![img](./ressources/Mécanisme%20de%20feedback2.jpg)
![img](./ressources/Mécanisme%20de%20feedback3.jpg)
![img](./ressources/Mécanisme%20de%20feedback4.jpg)
