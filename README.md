PARA FAZER RODAR O PROJETO, SIGA OS PASSOS ABAIXO: 


1 - Criar uma pasta na sua máquina para poder realizar o clone de forma organizada;

2 - Realizar o Clone do HTTPS do Repositório do GIT dentro da pasta criada no passo 1;

3 - Realizar a instalação das bibliotecas, pacotes, jupyter, python e demais;

4 - Após instalar as dependencias do projeto, execute todos os passos do arquivo mirroredhealth.ipynb para ele atualizar o modelo treinado;

5 - No terminal do VS CODE, execute o comando uvicorn main:app --reload, para poder ligar a API;

6 - Após a API do passo 5 estar rodando, abra a página formularioIA.html;

7 - Preencha os dados do formulário corretamente, e envie o mesmo;

8 - Pronto! 
      --> API recebe o formulário via JSON;
      --> API manda o formulário JSON para o modelo treinado da IA processar;
      --> API pega o resultado do modelo e exibe na tela de grafico.html 
