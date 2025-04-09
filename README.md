# BAC_Video-quality-assessment
Moja bakalárska práca - posudzovanie subjektívnej kvality videa podľa objektívnych parametrov (cez ML model), ktorý vyhodnotí subjektívnu kvalitu.
V práci je použitý jazyk Python, ktorého kód je realizovaný vo vývojovom prostredí/editore kódu Visual Studio Code. Na ML (machine learning) modely
sú použité knižnice ako Numpy, Pandas, Matplotlib, Scikit-Learn, Keras, PySide6.
Model používaný v aplikácii je v súbore model110b64b005.keras a pca a scaler v súboroch pc.pkl a scaler.pkl v tomto poradí.

Súbor user_interface.py je súbor s grafickým rozhraním, z ktorého spúšťa aplikácia - teda na jej použitie treba spustiť tento súbor (slúži ako "main"). Tiež obsahuje funkciu na vykreslenie MOS hodnoty oproti meniacej sa hodnote SSIM, VMAF a stratovosti do grafov.

Súbor input_data_parsing.py obsahuje funkcie na načítavanie dáť zo súborov first_session.csv a second_sessions.csv, čo zahŕňa aj funkcie na získanie číslených hodnôt zo scény, rozlíšenia a pod. Tiež obsahuje funkciu, ktorou sa trénuje model na týchto údajoch.

Súbor network_training.py obsahuje funkciu na spracovanie údajov pomocou metódy PCA (Principal component analysis) pred trénovaním dát funkciu na samotné trénovanie, v ktorej sa podľa parametrov zostaví model neurónovej siete a trénuje sa.

Súbor network_experiments.py obsahuje funkcie na testovanie siete - konfigurácie, aktivačných funkcií a na vykreslenie výsledkov trénovania do grafu.
