---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__(<p>__start__</p>)
        validar_entrada(validar_entrada)
        extrair_intencao(extrair_intencao)
        orientar_usuario(orientar_usuario)
        verificar_escopo(verificar_escopo)
        processar_solicitacoes(processar_solicitacoes)
        compor_resposta(compor_resposta)
        injetar_aviso(injetar_aviso)
        avaliar_saida(avaliar_saida)
        sintetizar_audio(sintetizar_audio)
        montar_resposta_final(montar_resposta_final)
        resposta_entrada_invalida(resposta_entrada_invalida)
        resposta_fora_de_escopo(resposta_fora_de_escopo)
        __end__(<p>__end__</p>)
        __start__ --> validar_entrada;
        validar_entrada --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc