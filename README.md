Projeto de Detecção de Mãos e Bolinha
Este projeto visa a detecção de mãos e a identificação de uma bolinha utilizando técnicas de visão computacional e Machine Learning. Desenvolvido em Python, o sistema combina a biblioteca MediaPipe para rastreamento de mãos e um modelo de aprendizado de máquina para prever a presença da bolinha em uma cena.

Descrição do Projeto
Objetivo: O principal objetivo é identificar a quantidade de dedos levantados e detectar se uma bolinha está presente, utilizando dados coletados em tempo real a partir de uma webcam.

Tecnologia:

Python: A linguagem de programação utilizada para implementar o projeto, escolhida por sua versatilidade e vasto suporte em bibliotecas de ciência de dados.
MediaPipe: Uma biblioteca de código aberto que facilita o rastreamento de mãos e a extração de características faciais.
Modelo de Machine Learning: Um modelo treinado que utiliza características das mãos para prever a presença da bolinha. O treinamento foi feito utilizando um conjunto de dados com as informações das mãos em diferentes posições.
Funcionalidades:

Contagem de Dedos: O sistema é capaz de contar quantos dedos estão levantados com base na posição dos marcos das mãos.
Detecção da Bolinha: Além da contagem de dedos, o sistema também é capaz de prever se a bolinha está próxima, usando um modelo treinado para classificar a presença da bolinha com base nas características extraídas das mãos.
Interface do Usuário: O projeto foi implementado em uma interface simples que mostra a contagem de dedos e o status da bolinha (se está "Pega" ou "Longe") em tempo real. A visualização é feita através de um feed de webcam, permitindo que o usuário interaja diretamente com o sistema.

Aplicações Futuras: Esse projeto tem potencial para ser expandido em várias áreas, como jogos interativos, interfaces de controle gestual e aplicações em robótica, onde a detecção precisa de mãos e objetos é crucial.
