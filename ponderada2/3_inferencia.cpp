#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

class Camada {
public:
    virtual std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& entrada) = 0;

    virtual ~Camada() {}
};

class CamadaConvolucional : public Camada {
private:
    int numFiltros;
    int tamanhoKernel;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::vector<double>>>> kernels; // Kernels: numFiltros x canaisEntrada x tamanhoKernel x tamanhoKernel

public:
    CamadaConvolucional(int numFiltros, int tamanhoKernel, int canaisEntrada, int stride = 1, int padding = 0) {
        this->numFiltros = numFiltros;
        this->tamanhoKernel = tamanhoKernel;
        this->stride = stride;
        this->padding = padding;
        inicializarKernels(canaisEntrada);
    }

    void inicializarKernels(int canaisEntrada) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        for (int k = 0; k < numFiltros; ++k) {
            std::vector<std::vector<std::vector<double>>> kernelPorCanal(canaisEntrada);
            for (int c = 0; c < canaisEntrada; ++c) {
                std::vector<std::vector<double>> kernel(tamanhoKernel, std::vector<double>(tamanhoKernel));
                for (int i = 0; i < tamanhoKernel; ++i) {
                    for (int j = 0; j < tamanhoKernel; ++j) {
                        kernel[i][j] = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
                    }
                }
                kernelPorCanal[c] = kernel;
            }
            kernels.push_back(kernelPorCanal);
        }
    }

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& entrada) override {
        int canaisEntrada = entrada.size();
        int linhasImagem = entrada[0].size();
        int colunasImagem = entrada[0][0].size();

        std::vector<std::vector<std::vector<double>>> imagemProcessada = entrada;
        if (padding > 0) {
            imagemProcessada = aplicarPadding(entrada, padding);
            linhasImagem += 2 * padding;
            colunasImagem += 2 * padding;
        }

        int linhasSaida = ((linhasImagem - tamanhoKernel) / stride) + 1;
        int colunasSaida = ((colunasImagem - tamanhoKernel) / stride) + 1;

        std::vector<std::vector<std::vector<double>>> saida(numFiltros,
            std::vector<std::vector<double>>(linhasSaida, std::vector<double>(colunasSaida, 0.0)));

        for (int f = 0; f < numFiltros; ++f) {
            for (int i = 0; i < linhasSaida; ++i) {
                for (int j = 0; j < colunasSaida; ++j) {
                    double soma = 0.0;
                    for (int c = 0; c < canaisEntrada; ++c) {
                        for (int m = 0; m < tamanhoKernel; ++m) {
                            for (int n = 0; n < tamanhoKernel; ++n) {
                                int x = i * stride + m;
                                int y = j * stride + n;
                                soma += imagemProcessada[c][x][y] * kernels[f][c][m][n];
                            }
                        }
                    }
                    saida[f][i][j] = soma;
                }
            }
        }

        return saida;
    }

    std::vector<std::vector<std::vector<double>>> aplicarPadding(const std::vector<std::vector<std::vector<double>>>& imagem, int padding) {
        int canais = imagem.size();
        int linhasOriginais = imagem[0].size();
        int colunasOriginais = imagem[0][0].size();
        int novasLinhas = linhasOriginais + 2 * padding;
        int novasColunas = colunasOriginais + 2 * padding;

        std::vector<std::vector<std::vector<double>>> imagemPadded(canais,
            std::vector<std::vector<double>>(novasLinhas, std::vector<double>(novasColunas, 0.0)));

        for (int c = 0; c < canais; ++c) {
            for (int i = 0; i < linhasOriginais; ++i) {
                for (int j = 0; j < colunasOriginais; ++j) {
                    imagemPadded[c][i + padding][j + padding] = imagem[c][i][j];
                }
            }
        }

        return imagemPadded;
    }
};

class CamadaDensa : public Camada {
private:
    int tamanhoEntrada;
    int tamanhoSaida;
    std::vector<std::vector<double>> pesos;
    std::vector<double> bias;               
    std::string funcaoAtivacao;

public:
    CamadaDensa(int tamanhoEntrada, int tamanhoSaida, const std::string& funcaoAtivacao) {
        this->tamanhoEntrada = tamanhoEntrada;
        this->tamanhoSaida = tamanhoSaida;
        this->funcaoAtivacao = funcaoAtivacao;
        inicializarPesos();
    }

    void inicializarPesos() {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        pesos.resize(tamanhoEntrada, std::vector<double>(tamanhoSaida));
        bias.resize(tamanhoSaida);

        for (int i = 0; i < tamanhoEntrada; ++i) {
            for (int j = 0; j < tamanhoSaida; ++j) {
                pesos[i][j] = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
            }
        }

        for (int j = 0; j < tamanhoSaida; ++j) {
            bias[j] = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
        }
    }

    double ativacao(double x) {
        if (funcaoAtivacao == "relu") {
            return std::max(0.0, x);
        } else if (funcaoAtivacao == "sigmoide") {
            return 1.0 / (1.0 + std::exp(-x));
        } else {
            return x;
        }
    }

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& entrada) override {
        std::vector<double> entradaFlattened = flatten(entrada);

        if (entradaFlattened.size() != static_cast<size_t>(tamanhoEntrada)) {
            std::cerr << "Erro: Tamanho da entrada não corresponde ao tamanho esperado pela camada densa." << std::endl;
            exit(1);
        }

        std::vector<double> saida(tamanhoSaida, 0.0);
        for (int j = 0; j < tamanhoSaida; ++j) {
            for (int i = 0; i < tamanhoEntrada; ++i) {
                saida[j] += entradaFlattened[i] * pesos[i][j];
            }
            saida[j] += bias[j];
            saida[j] = ativacao(saida[j]);
        }

        std::vector<std::vector<std::vector<double>>> saidaTensor(1, std::vector<std::vector<double>>(1, saida));

        return saidaTensor;
    }

    std::vector<double> flatten(const std::vector<std::vector<std::vector<double>>>& tensor) {
        std::vector<double> vetor;
        for (const auto& matriz : tensor) {
            for (const auto& linha : matriz) {
                for (double valor : linha) {
                    vetor.push_back(valor);
                }
            }
        }
        return vetor;
    }
};


class RedeNeural {
private:
    std::vector<Camada*> camadas;

public:
    ~RedeNeural() {
        for (Camada* camada : camadas) {
            delete camada;
        }
    }

    void adicionarCamada(Camada* camada) {
        camadas.push_back(camada);
    }

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& entrada) {
        std::vector<std::vector<std::vector<double>>> saida = entrada;
        for (Camada* camada : camadas) {
            saida = camada->forward(saida);
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
    RedeNeural rede;

    int numFiltros = 8;
    int tamanhoKernel = 3;
    int stride = 1;
    int padding = 1;
    int canaisEntrada = 1;

    CamadaConvolucional* camadaConv = new CamadaConvolucional(numFiltros, tamanhoKernel, canaisEntrada, stride, padding);
    rede.adicionarCamada(camadaConv);

    int tamanhoEntradaDensa = ((28 + 2 * padding - tamanhoKernel) / stride + 1) * ((28 + 2 * padding - tamanhoKernel) / stride + 1) * numFiltros;
    int tamanhoSaidaDensa = 10;
    CamadaDensa* camadaDensa = new CamadaDensa(tamanhoEntradaDensa, tamanhoSaidaDensa, "sigmoide");
    rede.adicionarCamada(camadaDensa);

    std::vector<std::vector<std::vector<double>>> entrada(1); 
    entrada[0] = criarImagemExemplo(28, 28);

    std::vector<std::vector<std::vector<double>>> saida = rede.forward(entrada);

    std::cout << "Saída da rede neural:" << std::endl;
    for (const auto& matriz : saida) {
        for (const auto& linha : matriz) {
            for (double valor : linha) {
                std::cout << valor << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
