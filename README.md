# Dashboard de Curtailment — Mauriti vs Benchmark CE

Estudo de constrained-off do **Complexo Fotovoltaico Mauriti** (PowerChina), com
benchmarking contra ativos do Ceará. Atualizado a partir de dados públicos do
**ONS** e **CCEE**.

> **Acesse a versão publicada:** `https://SEU-USUARIO.github.io/SEU-REPO/`
> *(substitua pelo seu usuário/repo após o setup do GitHub Pages descrito abaixo)*

---

## O que tem no dashboard

1. **Tracker do mês corrente** — barras diárias estimada vs realizada + linha de
   CF% diário, com referência do CF% médio dos últimos 90 dias. Atualizado
   semanalmente pelo workflow.
2. **Hero + KPIs do período** — energia cortada, curtailment factor, receita
   perdida estimada a PLD, fração potencialmente ressarcível (REL+CNF).
3. **Série temporal** — geração estimada vs realizada com curtailment destacado.
4. **Razões do corte** — donut + stacked bar mensal com REL/CNF/ENE/PAR.
5. **Mauriti vs benchmark CE** — uma linha por grupo de ativos (Abaiara,
   Lins, Calcário, Alex, Banabuiú, Serra do Mato, Sol do Futuro), com Mauriti
   destacada em vermelho-tijolo grosso.
6. **Heatmap hora × dia** — onde no dia os cortes acontecem (revela se a
   restrição é sistêmica do submercado NE ou local).

---

## Uso local (rápido)

Requer **Python 3.10+**.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate              # Windows
pip install -r requirements.txt
python gerar_dashboard_curtailment.py
# Abre public/index.html no navegador
```

**Tempo estimado da primeira execução**: 5–10 minutos (baixa ~10 meses × 4
datasets do ONS + 2 anos de PLD da CCEE). Execuções seguintes são bem mais
rápidas porque só re-baixam os 3 meses mais recentes — o resto fica em
`./cache/`.

---

## Setup de publicação automática semanal (GitHub Pages)

A configuração abaixo faz o dashboard **rodar toda segunda-feira às 9h da
manhã (BRT)** e publicar uma URL pública. Tudo de graça, na nuvem da
Microsoft.

### Passo 1 — Criar o repositório

1. Vá em [github.com/new](https://github.com/new)
2. Nome do repositório: ex. `mauriti-curtailment` (pode ser **privado** — Pages
   funciona em repos privados na conta Pro/Team; em conta Free precisa ser
   público)
3. Marque "Add a README" pra criar o repo já com um commit inicial
4. Crie o repositório

### Passo 2 — Subir os arquivos

Em qualquer terminal local, depois de clonar o repo:

```bash
git clone https://github.com/SEU-USUARIO/mauriti-curtailment.git
cd mauriti-curtailment

# Copia os 4 arquivos pra dentro do repo:
#   - gerar_dashboard_curtailment.py
#   - requirements.txt
#   - README.md  (este arquivo)
#   - atualiza-dashboard.yml  →  mover para .github/workflows/

mkdir -p .github/workflows
mv atualiza-dashboard.yml .github/workflows/

# .gitignore - importante pra não subir cache pesado
cat > .gitignore <<EOF
__pycache__/
*.pyc
*.part
.venv/
.env
cache/
public/
EOF

git add .
git commit -m "Setup dashboard de curtailment Mauriti"
git push origin main
```

### Passo 3 — Ativar o GitHub Pages

1. No GitHub, abra o repositório
2. Vai em **Settings → Pages** (menu lateral esquerdo)
3. Em **Source**, selecione **Deploy from a branch**
4. Em **Branch**, escolha **`gh-pages`** com pasta **`/ (root)`**
   - **Atenção**: a branch `gh-pages` ainda não existe — ela será criada na
     primeira execução do workflow (próximo passo). Por enquanto vai aparecer
     vazia. Tudo bem.
5. Clica em **Save**

### Passo 4 — Rodar o workflow pela primeira vez

1. Abre a aba **Actions** do repositório
2. Na lista da esquerda, clica em **"Atualiza Dashboard Curtailment Mauriti"**
3. Clica em **Run workflow** (botão azul à direita) → **Run workflow**
4. Aguarda ~5–10 minutos pra ele baixar tudo do ONS e gerar o HTML
5. Quando ficar verde, volta em **Settings → Pages** — agora a branch
   `gh-pages` existe e o GitHub Pages mostra a URL pública (algo como
   `https://SEU-USUARIO.github.io/mauriti-curtailment/`)

### Passo 5 — Pronto

Daqui em diante, **toda segunda às 9h BRT** o workflow roda sozinho, baixa os
dados novos (ou re-baixa os 3 últimos meses pra capturar revisões do ONS),
gera o HTML novo, e publica na URL. Você só visita o link.

Se quiser forçar uma atualização extra a qualquer momento, basta ir em
**Actions → Run workflow** novamente.

---

## Como os dados são baixados

O script faz tudo automaticamente:

| Fonte | O que baixa | Quando re-baixa |
|---|---|---|
| ONS S3 | Constrained-off detalhe **eólica** (CSV mês a mês) | Sempre os 3 meses mais recentes |
| ONS S3 | Constrained-off detalhe **solar** (CSV mês a mês) | Sempre os 3 meses mais recentes |
| ONS S3 | Constrained-off consolidado **eólica** (CSV mês a mês, com razão do corte) | Sempre os 3 meses mais recentes |
| ONS S3 | Constrained-off consolidado **solar** (CSV mês a mês, com razão do corte) | Sempre os 3 meses mais recentes |
| CCEE | PLD horário (CSV ano a ano) | Sempre o ano corrente |

Tudo cacheado em `./cache/` (no GitHub Actions, o cache persiste entre runs
via `actions/cache`).

A re-baixa dos 3 meses recentes é importante porque o ONS aplica
**consistência recorrente** — dados publicados podem ser revisados por até
30–60 dias após a publicação inicial. Mantemos o cache fresco nessa janela.

---

## Como ajustar a lista de usinas do benchmark

No topo de `gerar_dashboard_curtailment.py`, edite o bloco
`CONFIG["benchmark_groups"]`:

```python
"benchmark_groups": [
    {"label": "Abaiara 230 kV",  "match": "ABAIARA",       "fonte": "UFV"},
    {"label": "Lins",            "match": "LINS",          "fonte": "UFV"},
    {"label": "Calcario",        "match": "CALCARIO",      "fonte": "UFV"},
    {"label": "Alex",            "match": "ALEX",          "fonte": "UFV"},
    {"label": "Banabuiu",        "match": "BANABUIU",      "fonte": "EOL"},
    {"label": "Serra do Mato",   "match": "SERRA DO MATO", "fonte": "EOL"},
    {"label": "Sol do Futuro",   "match": "SOL DO FUTURO", "fonte": "UFV"},
],
```

Como funciona o `match`:
- Substring **case-insensitive sem acento** que tem que aparecer em `nom_usina`
- "Sol do Futuro" → casa `UFV SOL DO FUTURO 1`, `UFV SOL DO FUTURO 2`,
  `UFV SOL DO FUTURO 3` → todos viram **uma linha agregada** no chart
- `fonte` (`"UFV"` ou `"EOL"`) é só pra colorir a linha (UFV = bege/marrom,
  EOL = azul-acinzentado)

**Ao rodar o script, ele imprime no console:**

```
Grupos benchmark configurados:
  + Abaiara 230 kV    : 4 usina(s) -> UFV ABAIARA 1, UFV ABAIARA 2, ...
  + Lins              : 1 usina(s) -> UFV LINS
  ! Calcario          : NENHUMA usina encontrada (match='CALCARIO')
```

Se aparecer `! NENHUMA`, o nome no ONS é diferente. Algumas opções:
- Tente abreviar o `match` (ex: `"CALC"` em vez de `"CALCARIO"`)
- Confira o nome exato no portal ONS:
  https://dados.ons.org.br/dataset/restricao_coff_fotovoltaica
- Se quiser ser exato e robusto, pode trocar por CEG (precisa pequena
  alteração na função `selecionar_grupos`).

---

## Outros ajustes comuns

| Quero mudar... | Onde |
|---|---|
| Período de análise | `CONFIG["data_inicio"]` e `data_fim` (None = hoje) |
| Submercado | `CONFIG["submercado"]` ("NE", "SE", "S", "N") |
| Quantos meses re-baixar | `CONFIG["refresh_recent_n"]` (default 3) |
| Frequência do workflow | linha `cron:` em `.github/workflows/atualiza-dashboard.yml` |
| Pasta de saída | `CONFIG["output_html"]` |

---

## Sobre as razões do corte (REN ANEEL 1.030/2022)

| Código | Descrição | Tipicamente ressarcível? |
|---|---|---|
| **REL** | Indisponibilidade externa (elétrica) | ✅ Sim |
| **CNF** | Confiabilidade | ✅ Sim |
| **ENE** | Energético | ❌ Em geral não |
| **PAR** | Parecer de acesso | ❌ Em geral não |

O KPI "Potencial ressarcível" do dashboard soma só REL+CNF — é uma estimativa
direcional. A apuração final depende da modalidade da usina (Tipo I, II-B,
II-C) e dos termos do PPA.

---

## Resolução de problemas

**O workflow falhou na primeira execução**
Verifique em **Settings → Actions → General → Workflow permissions** se está
marcado "Read and write permissions". Salve e rode de novo.

**Algumas usinas do benchmark não aparecem**
Veja no log da execução do workflow (aba Actions → run → step "Gera o
dashboard") quais grupos retornaram `! NENHUMA usina encontrada` e ajuste o
`match` no script (veja seção "Como ajustar a lista de usinas" acima).

**O HTML aparece sem dados / branco**
Provavelmente o ONS ainda não publicou o mês corrente no dataset detalhado.
Confira em https://dados.ons.org.br/dataset/restricao_coff_fotovoltaica_detail
qual o último mês disponível e ajuste `data_fim` no CONFIG se necessário.

**Cache do GitHub Actions ficou velho**
Em **Actions → Caches** você pode deletar o cache manualmente. A próxima run
vai re-baixar tudo do zero.

---

## Estrutura do projeto

```
mauriti-curtailment/
├─ .github/
│  └─ workflows/
│     └─ atualiza-dashboard.yml      # roda toda segunda 9h BRT
├─ gerar_dashboard_curtailment.py    # script principal
├─ requirements.txt                  # dependencias Python
├─ README.md                         # este arquivo
├─ .gitignore                        # ignora cache/, public/, .venv/
├─ cache/                            # gerado: dados ONS+CCEE (gitignore)
└─ public/
   └─ index.html                     # gerado: dashboard final (gitignore)
```

---

## Próximos passos sugeridos

- Cruzar com **PPA contratado** das usinas Mauriti (custo real do MWh
  perdido = `(PPA − PLD) × MWh` quando PPA > PLD)
- Comparar com **período homólogo** do ano anterior
- Adicionar **alerta automático** (Slack/e-mail) se CF% do dia anterior
  ultrapassar X%
- Versão **PDF/PPT** dos slides principais pra distribuição em diretoria

Pra qualquer ajuste, basta editar o script, dar push, e o GitHub Actions
reprocessa tudo no próximo agendamento (ou rode `Run workflow` manual).
