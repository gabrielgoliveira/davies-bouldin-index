# dunn-index

### Executando o projeto :

Compilação : `nvcc db.cu -o db`
Executar   : `./db`


### Semantica do dataset :

* Linha 1: A linha deve ter 2 valores, o primeiro é o numero de clusters e o segundo é o numero de feature por cluster
* Linha 2: Representa a quantidade de pontos de cada cluster, ou seja, se na primeira linha foi informado 10 clusters, então teremos 10 valores na segunda linha, onde cada valor é a quantidade de dados dentro do cluster.
* Demais linhas: Cada linha representa um ponto do cluster, as linhas estão ordenadas por cluster.