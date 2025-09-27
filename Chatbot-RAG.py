"""
Chatbot RAG (Retrieval Augmented Generation) - Archivo Principal

Este es el punto de entrada principal del chatbot que integra:
- DocumentIndexer para indexación de documentos
- RAGChatbot para generación de respuestas
- Funciones de utilidad para carga de documentos
- Interfaz de línea de comandos interactiva
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

# Importar nuestros módulos
from document_indexer import DocumentIndexer
from rag_chatbot import RAGChatbot
from utils import (
    load_document, 
    get_supported_file_types, 
    get_file_type_description,
    validate_file_for_processing
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = os.getenv("CHROMA_DB_PERSIST_DIR", "./chroma_db")

# Verificar que la API key de OpenAI esté configurada
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY no está configurada en el archivo .env")


def main():
    """Función principal con interfaz de línea de comandos."""
    print("="*60)
    print("🤖 CHATBOT RAG - Retrieval Augmented Generation")
    print("="*60)
    print("Powered by OpenAI + ChromaDB")
    print()
    
    try:
        # Inicializar el chatbot
        chatbot = RAGChatbot()
        print("✅ Chatbot inicializado correctamente")
        
        # Mostrar información de la base de datos
        db_info = chatbot.indexer.get_collection_info()
        print(f"📊 Documentos en la base de datos: {db_info['total_documents']}")
        print()
        
        while True:
            print("\n" + "="*50)
            print("MENÚ PRINCIPAL")
            print("="*50)
            print("1. 📄 Indexar documento(s)")
            print("2. 💬 Hacer consulta al chatbot")
            print("3. 📊 Ver información de la base de datos")
            print("4. 🔍 Ver tipos de archivo soportados")
            print("5. 🧹 Limpiar historial de conversación")
            print("6. 📝 Ver resumen de conversación")
            print("7. 💡 Obtener sugerencias de preguntas")
            print("8. 📋 Listar documentos disponibles")
            print("9. ❌ Salir")
            print()
            
            try:
                opcion = input("Selecciona una opción (1-9): ").strip()
                
                if opcion == "1":
                    indexar_documentos(chatbot.indexer)
                
                elif opcion == "2":
                    hacer_consulta(chatbot)
                
                elif opcion == "3":
                    mostrar_info_db(chatbot.indexer)
                
                elif opcion == "4":
                    mostrar_tipos_soportados()
                
                elif opcion == "5":
                    chatbot.clear_history()
                    print("✅ Historial de conversación limpiado")
                
                elif opcion == "6":
                    mostrar_resumen_conversacion(chatbot)
                
                elif opcion == "7":
                    mostrar_sugerencias(chatbot)
                
                elif opcion == "8":
                    listar_documentos(chatbot)
                
                elif opcion == "9":
                    print("👋 ¡Hasta luego!")
                    break
                
                else:
                    print("❌ Opción inválida. Por favor selecciona 1-9.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error inesperado: {e}")
                logger.error(f"Error en menú principal: {e}")
    
    except Exception as e:
        print(f"❌ Error inicializando el chatbot: {e}")
        logger.error(f"Error en main: {e}")


def indexar_documentos(indexer: DocumentIndexer):
    """Interfaz para indexar documentos."""
    print("\n" + "="*40)
    print("📄 INDEXAR DOCUMENTOS")
    print("="*40)
    
    while True:
        print("\nOpciones:")
        print("1. Indexar un archivo específico")
        print("2. Indexar todos los archivos de una carpeta")
        print("3. Volver al menú principal")
        
        opcion = input("\nSelecciona una opción (1-3): ").strip()
        
        if opcion == "1":
            archivo = input("Ingresa la ruta del archivo: ").strip()
            if archivo:
                indexar_archivo_individual(indexer, archivo)
            else:
                print("❌ Ruta no válida")
                
        elif opcion == "2":
            carpeta = input("Ingresa la ruta de la carpeta(documentos): ").strip()
            if carpeta:
                indexar_carpeta(indexer, carpeta)
            else:
                indexar_carpeta(indexer, "./documentos")
                
        elif opcion == "3":
            break
            
        else:
            print("❌ Opción inválida")


def indexar_archivo_individual(indexer: DocumentIndexer, archivo_path: str):
    """Indexa un archivo individual."""
    try:
        print(f"📄 Procesando: {archivo_path}")
        
        # Validar archivo
        validation = validate_file_for_processing(archivo_path)
        if not validation["valid"]:
            print(f"❌ {validation['error']}")
            return
        
        if validation.get("warning"):
            print(f"⚠️ {validation['warning']}")
            confirmar = input("¿Continuar? (s/n): ").strip().lower()
            if confirmar != 's':
                print("❌ Operación cancelada")
                return
        
        # Cargar el documento
        contenido, metadata = load_document(archivo_path)
        print(f"✅ Archivo cargado: {metadata['word_count']} palabras, {metadata['content_length']} caracteres")
        
        # Indexar
        filename = Path(archivo_path).name
        chunks_indexados = indexer.index_document(contenido, filename, metadata)
        
        if chunks_indexados > 0:
            print(f"✅ Documento indexado exitosamente: {chunks_indexados} chunks creados")
        else:
            print("❌ No se pudo indexar el documento")
            
    except Exception as e:
        print(f"❌ Error procesando el archivo: {e}")
        logger.error(f"Error indexando archivo {archivo_path}: {e}")


def indexar_carpeta(indexer: DocumentIndexer, carpeta_path: str):
    """Indexa todos los archivos soportados de una carpeta."""
    try:
        carpeta = Path(carpeta_path)
        if not carpeta.exists():
            print("❌ La carpeta no existe")
            return
            
        if not carpeta.is_dir():
            print("❌ La ruta no es una carpeta")
            return
        
        # Buscar archivos soportados
        tipos_soportados = get_supported_file_types()
        archivos_encontrados = []
        
        for tipo in tipos_soportados:
            archivos_encontrados.extend(list(carpeta.glob(f"*{tipo}")))
        
        if not archivos_encontrados:
            print(f"❌ No se encontraron archivos soportados en {carpeta_path}")
            return
        
        print(f"📁 Encontrados {len(archivos_encontrados)} archivo(s) para indexar:")
        for archivo in archivos_encontrados:
            print(f"  - {archivo.name}")
        
        confirmar = input("\n¿Proceder con la indexación? (s/n): ").strip().lower()
        if confirmar != 's':
            print("❌ Indexación cancelada")
            return
        
        # Indexar archivos
        exitosos = 0
        fallidos = 0
        
        for archivo in archivos_encontrados:
            try:
                print(f"\n📄 Procesando: {archivo.name}")
                contenido, metadata = load_document(str(archivo))
                chunks = indexer.index_document(contenido, archivo.name, metadata)
                
                if chunks > 0:
                    print(f"✅ {archivo.name}: {chunks} chunks indexados")
                    exitosos += 1
                else:
                    print(f"❌ {archivo.name}: Error en la indexación")
                    fallidos += 1
                    
            except Exception as e:
                print(f"❌ {archivo.name}: Error - {str(e)}")
                fallidos += 1
        
        print(f"\n📊 Resumen de indexación:")
        print(f"  ✅ Exitosos: {exitosos}")
        print(f"  ❌ Fallidos: {fallidos}")
        
    except Exception as e:
        print(f"❌ Error procesando la carpeta: {e}")
        logger.error(f"Error indexando carpeta {carpeta_path}: {e}")


def hacer_consulta(chatbot: RAGChatbot):
    """Interfaz para hacer consultas al chatbot."""
    print("\n" + "="*40)
    print("💬 CONSULTAS AL CHATBOT")
    print("="*40)
    print("Escribe 'salir' para volver al menú principal")
    print()
    
    while True:
        try:
            pregunta = input("🤔 Tu pregunta: ").strip()
            
            if pregunta.lower() in ['salir', 'exit', 'quit']:
                break
            
            if not pregunta:
                print("❌ Por favor ingresa una pregunta")
                continue
            
            print("\n🤖 Procesando...")
            respuesta = chatbot.generate_response(pregunta)
            
            print("\n" + "="*50)
            print("🤖 RESPUESTA:")
            print("="*50)
            print(respuesta)
            print("="*50)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Volviendo al menú principal...")
            break
        except Exception as e:
            print(f"❌ Error procesando la consulta: {e}")
            logger.error(f"Error en consulta: {e}")


def mostrar_info_db(indexer: DocumentIndexer):
    """Muestra información de la base de datos."""
    print("\n" + "="*40)
    print("📊 INFORMACIÓN DE LA BASE DE DATOS")
    print("="*40)
    
    try:
        info = indexer.get_collection_info()
        print(f"📄 Total de documentos indexados: {info['total_documents']}")
        print(f"📚 Nombre de la colección: {info['collection_name']}")
        print(f"💾 Directorio de persistencia: {PERSIST_DIR}")
        
        if info['total_documents'] == 0:
            print("\n💡 La base de datos está vacía. Indexa algunos documentos primero.")
        
    except Exception as e:
        print(f"❌ Error obteniendo información: {e}")
        logger.error(f"Error obteniendo info de DB: {e}")


def mostrar_tipos_soportados():
    """Muestra los tipos de archivo soportados."""
    print("\n" + "="*40)
    print("🔍 TIPOS DE ARCHIVO SOPORTADOS")
    print("="*40)
    
    tipos = get_supported_file_types()
    for tipo in tipos:
        descripcion = get_file_type_description(tipo)
        print(f"📄 {tipo.upper()} - {descripcion}")


def mostrar_resumen_conversacion(chatbot: RAGChatbot):
    """Muestra el resumen de la conversación actual."""
    print("\n" + "="*40)
    print("📝 RESUMEN DE CONVERSACIÓN")
    print("="*40)
    
    try:
        resumen = chatbot.get_conversation_summary()
        
        print(f"💬 Total de intercambios: {resumen['total_exchanges']}")
        
        if resumen['last_query']:
            print(f"❓ Última pregunta: {resumen['last_query'][:100]}...")
        else:
            print("❓ Última pregunta: Ninguna")
        
        if resumen['documents_referenced']:
            print(f"📚 Documentos referenciados:")
            for doc in resumen['documents_referenced']:
                if doc:  # Evitar valores None
                    print(f"  - {doc}")
        else:
            print("📚 Documentos referenciados: Ninguno")
            
    except Exception as e:
        print(f"❌ Error obteniendo resumen: {e}")
        logger.error(f"Error obteniendo resumen: {e}")


def mostrar_sugerencias(chatbot: RAGChatbot):
    """Muestra sugerencias de preguntas."""
    print("\n" + "="*40)
    print("💡 SUGERENCIAS DE PREGUNTAS")
    print("="*40)
    
    try:
        print("🤖 Generando sugerencias basadas en los documentos...")
        sugerencias = chatbot.suggest_questions()
        
        if sugerencias:
            print("\n💡 Preguntas sugeridas:")
            for i, pregunta in enumerate(sugerencias, 1):
                print(f"{i}. {pregunta}")
        else:
            print("❌ No se pudieron generar sugerencias")
            
    except Exception as e:
        print(f"❌ Error generando sugerencias: {e}")
        logger.error(f"Error generando sugerencias: {e}")


def listar_documentos(chatbot: RAGChatbot):
    """Lista todos los documentos disponibles."""
    print("\n" + "="*40)
    print("📋 DOCUMENTOS DISPONIBLES")
    print("="*40)
    
    try:
        documentos = chatbot.list_available_documents()
        
        if documentos:
            print(f"📚 Documentos en la base de datos ({len(documentos)}):")
            for i, doc in enumerate(documentos, 1):
                print(f"{i}. {doc}")
        else:
            print("📭 No hay documentos indexados")
            print("\n💡 Usa la opción 1 del menú principal para indexar documentos")
            
    except Exception as e:
        print(f"❌ Error listando documentos: {e}")
        logger.error(f"Error listando documentos: {e}")


if __name__ == "__main__":
    main()