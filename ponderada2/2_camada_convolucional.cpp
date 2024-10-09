#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

class CamadaConvolucional {
private:
    int numFiltros;
    int tamanhoKernel;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<double>>> kernels; 

public:
    CamadaConvolucional(int numFiltros, int tamanhoKernel, int stride = 1, int padding = 0) {
        this->numFiltros = numFiltros;
        this->tamanhoKernel = tamanhoKernel;
        this->stride = stride;
        this->padding = padding;
        inicializarKernels();
    }

    void inicializarKernels() {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        for (int k = 0; k < numFiltros; ++k) {
            std::vector<std::vector<double>> kernel(tamanhoKernel, std::vector<double>(tamanhoKernel));
            for (int i = 0; i < tamanhoKernel; ++i) {
                for (int j = 0; j < tamanhoKernel; ++j) {
                    kernel[i][j] = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
                }
            }
            kernels.push_back(kernel);
        }
    }

    std::vector<std::vector<double>> aplicarPadding(const std::vector<std::vector<double>>& imagem) {
        int linhasOriginais = imagem.size();
        int colunasOriginais = imagem[0].size();
        int novasLinhas = linhasOriginais + 2 * padding;
        int novasColunas = colunasOriginais + 2 * padding;

        std::vector<std::vector<double>> imagemPadded(novasLinhas, std::vector<double>(novasColunas, 0.0));

        for (int i = 0; i < linhasOriginais; ++i) {
            for (int j = 0; j < colunasOriginais; ++j) {
                imagemPadded[i + padding][j + padding] = imagem[i][j];
            }
        }

        return imagemPadded;
    }

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<double>>& imagem) {
        std::vector<std::vector<double>> imagemProcessada = imagem;
        if (padding > 0) {
            imagemProcessada = aplicarPadding(imagem);
        }

        int linhasImagem = imagemProcessada.size();
        int colunasImagem = imagemProcessada[0].size();

        int linhasSaida = ((linhasImagem - tamanhoKernel) / stride) + 1;
        int colunasSaida = ((colunasImagem - tamanhoKernel) / stride) + 1;

        std::vector<std::vector<std::vector<double>>> saida(numFiltros,
            std::vector<std::vector<double>>(linhasSaida, std::vector<double>(colunasSaida, 0.0)));

        for (int k = 0; k < numFiltros; ++k) {
            const std::vector<std::vector<double>>& kernel = kernels[k];

            for (int i = 0; i < linhasSaida; ++i) {
                for (int j = 0; j < colunasSaida; ++j) {
                    double soma = 0.0;
                    for (int m = 0; m < tamanhoKernel; ++m) {
                        for (int n = 0; n < tamanhoKernel; ++n) {
                            int x = i * stride + m;
                            int y = j * stride + n;
                            soma += imagemProcessada[x][y] * kernel[m][n];
                        }
                    }
                    saida[k][i][j] = soma;
                }
            }
        }

        return saida;
    }
};


std::vector<std::vector<double>> criarImagemExemplo(int linhas, int colunas) {
    std::vector<std::vector<double>> imagem(linhas, std::vector<double>(colunas, 0.0));

    for (int i = 0; i < linhas; ++i) {
        for (int j = 0; j < colunas; ++j) {
            imagem[i][j] = static_cast<double>((i + j) % 256);
        }
    }

    return imagem;
}



int main() {
    int linhasImagem = 28;
    int colunasImagem = 28;

    std::vector<std::vector<double>> imagem = criarImagemExemplo(linhasImagem, colunasImagem);

    int numFiltros = 4;   
    int tamanhoKernel = 3;
    int stride = 1;       
    int padding = 1;      

    CamadaConvolucional camadaConv(numFiltros, tamanhoKernel, stride, padding);

    std::vector<std::vector<std::vector<double>>> mapasCaracteristicas = camadaConv.forward(imagem);

    std::cout << "Mapas de Características (Resultado da Convolução):" << std::endl;
    for (int k = 0; k < numFiltros; ++k) {
        std::cout << "Mapa de Características " << k << ":" << std::endl;
        for (size_t i = 0; i < mapasCaracteristicas[k].size(); ++i) {
            for (size_t j = 0; j < mapasCaracteristicas[k][i].size(); ++j) {
                std::cout << mapasCaracteristicas[k][i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}