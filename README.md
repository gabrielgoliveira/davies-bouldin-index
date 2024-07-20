# dunn-index

### Executando o projeto :

Compilação : `nvcc db.cu -o db`
Executar   : `./db`


### Semantica do dataset :

* Linha 1: A linha deve ter 2 valores, o primeiro é o numero de clusters e o segundo é o numero de feature por cluster
* Linha 2: Representa a quantidade de pontos de cada cluster, ou seja, se na primeira linha foi informado 10 clusters, então teremos 10 valores na segunda linha, onde cada valor é a quantidade de dados dentro do cluster.
* Demais linhas: Cada linha representa um ponto do cluster, as linhas estão ordenadas por cluster.



/*
    float *last_cluster = d_clusters[9];
    int size_last_cluster = size_clusters[9];

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((size_last_cluster + threadsPerBlock.x - 1) / threadsPerBlock.x,  (n_feat + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cuda_print_matrix<<<numBlocks,  threadsPerBlock>>>(last_cluster, size_last_cluster, n_feat);
    cudaDeviceSynchronize();
*/