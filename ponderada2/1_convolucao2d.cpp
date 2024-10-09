#include <iostream>
#include <vector>

std::vector<std::vector<double>> convolucao2D(
    const std::vector<std::vector<double>>& imagem,
    const std::vector<std::vector<double>>& kernel)
{
    int linhasImagem = imagem.size();
    int colunasImagem = imagem[0].size();
    int linhasKernel = kernel.size();
    int colunasKernel = kernel[0].size();

    int linhasSaida = linhasImagem - linhasKernel + 1;
    int colunasSaida = colunasImagem - colunasKernel + 1;

    std::vector<std::vector<double>> saida(linhasSaida, std::vector<double>(colunasSaida, 0.0));

    for (int i = 0; i < linhasSaida; ++i) {
        for (int j = 0; j < colunasSaida; ++j) {
            double soma = 0.0;
            for (int m = 0; m < linhasKernel; ++m) {
                for (int n = 0; n < colunasKernel; ++n) {
                    soma += imagem[i + m][j + n] * kernel[m][n];
                }
            }
            saida[i][j] = soma;
        }
    }

    return saida;
}

std::vector<std::vector<double>> criarImagemExemplo(int linhas, int colunas) {
    std::vector<std::vector<double>> imagem(linhas, std::vector<double>(colunas, 0.0));

    for (int i = 0; i < linhas; ++i) {
        for (int j = 0; j < colunas; ++j) {
            imagem[i][j] = static_cast<double>((i + j) % 256);
        }
    }

    return imagem;
}

std::vector<std::vector<double>> criarKernelExemplo() {
    return {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };
}

int main() {
    int linhasImagem = 28;
    int colunasImagem = 28;
    std::vector<std::vector<double>> imagem = criarImagemExemplo(linhasImagem, colunasImagem);

    std::vector<std::vector<double>> kernel = criarKernelExemplo();

    std::vector<std::vector<double>> mapaCaracteristicas = convolucao2D(imagem, kernel);

    std::cout << "Mapa de Características (Resultado da Convolução):" << std::endl;
    for (size_t i = 0; i < mapaCaracteristicas.size(); ++i) {
        for (size_t j = 0; j < mapaCaracteristicas[i].size(); ++j) {
            std::cout << mapaCaracteristicas[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
