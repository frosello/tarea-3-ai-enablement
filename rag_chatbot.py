"""
RAGChatbot - Chatbot con Retrieval Augmented Generation.

Esta clase se encarga de:
- Generar respuestas usando documentos indexados como contexto
- Mantener historial de conversación
- Integrar búsqueda semántica con generación de texto
- Gestionar el contexto de la conversación
"""

import os
import logging
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

from document_indexer import DocumentIndexer

# Configurar logging
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verificar que la API key de OpenAI esté configurada
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY no está configurada en el archivo .env")


class RAGChatbot:
    """Chatbot con Retrieval Augmented Generation usando OpenAI y ChromaDB."""
    
    def __init__(self, collection_name: str = "documents"):
        """
        Inicializa el RAGChatbot.
        
        Args:
            collection_name (str): Nombre de la colección en ChromaDB
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.indexer = DocumentIndexer(collection_name)
        self.conversation_history = []
    
    def generate_response(self, query: str, max_history: int = 5, n_results: int = 3, 
                         model: str = "gpt-3.5-turbo", temperature: float = 0.3) -> str:
        """
        Genera una respuesta usando RAG.
        
        Args:
            query (str): Pregunta del usuario
            max_history (int): Máximo número de intercambios previos a incluir
            n_results (int): Número de documentos relevantes a buscar
            model (str): Modelo de OpenAI a utilizar
            temperature (float): Temperatura para la generación (creatividad)
            
        Returns:
            str: Respuesta generada por el chatbot
        """
        try:
            # Buscar documentos relevantes
            relevant_docs = self.indexer.search_documents(query, n_results=n_results)
            
            # Construir contexto con documentos relevantes
            context_parts = []
            for doc in relevant_docs:
                context_parts.append(f"Documento: {doc['metadata'].get('filename', 'Desconocido')}")
                context_parts.append(f"Contenido: {doc['content']}")
                context_parts.append("---")
            
            context = "\n".join(context_parts) if context_parts else "No se encontraron documentos relevantes."
            
            # Obtener historial reciente de conversación
            recent_history = self.conversation_history[-max_history:] if self.conversation_history else []
            
            # Construir el prompt
            messages = [
                {
                    "role": "system",
                    "content": """Eres un asistente útil que responde preguntas basándose en los documentos proporcionados. 
                    
Instrucciones:
1. Usa ÚNICAMENTE la información de los documentos proporcionados para responder
2. Si la información no está en los documentos, di claramente que no tienes esa información
3. Cita el nombre del documento cuando uses información específica de él
4. Sé preciso y conciso en tus respuestas
5. Si hay múltiples documentos relevantes, integra la información de manera coherente
6. Si no hay documentos relevantes, indica que no puedes responder basándote en los documentos disponibles"""
                }
            ]
            
            # Agregar historial de conversación
            for hist_item in recent_history:
                messages.append({"role": "user", "content": hist_item["query"]})
                messages.append({"role": "assistant", "content": hist_item["response"]})
            
            # Agregar la consulta actual con contexto
            user_message = f"""Contexto de documentos:
{context}

Pregunta: {query}"""
            
            messages.append({"role": "user", "content": user_message})
            
            # Generar respuesta usando GPT
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            
            # Guardar en el historial
            self.conversation_history.append({
                "query": query,
                "response": answer,
                "relevant_docs": [doc['metadata'].get('filename') for doc in relevant_docs],
                "timestamp": self._get_current_timestamp()
            })
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return f"Lo siento, hubo un error al generar la respuesta: {str(e)}"
    
    def clear_history(self):
        """Limpia el historial de conversación."""
        self.conversation_history = []
        logger.info("Historial de conversación limpiado")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de la conversación actual.
        
        Returns:
            Dict[str, Any]: Resumen con estadísticas de la conversación
        """
        return {
            "total_exchanges": len(self.conversation_history),
            "last_query": self.conversation_history[-1]["query"] if self.conversation_history else None,
            "documents_referenced": list(set([
                doc for exchange in self.conversation_history 
                for doc in exchange.get("relevant_docs", [])
                if doc  # Filtrar valores None
            ]))
        }
    
    def get_conversation_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de conversación.
        
        Args:
            limit (int, optional): Límite de intercambios a retornar
            
        Returns:
            List[Dict[str, Any]]: Lista del historial de conversación
        """
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()
    
    def export_conversation(self) -> str:
        """
        Exporta la conversación en formato legible.
        
        Returns:
            str: Conversación formateada como texto
        """
        if not self.conversation_history:
            return "No hay conversación para exportar."
        
        export_lines = []
        export_lines.append("=== HISTORIAL DE CONVERSACIÓN ===\n")
        
        for i, exchange in enumerate(self.conversation_history, 1):
            export_lines.append(f"--- Intercambio {i} ---")
            export_lines.append(f"Timestamp: {exchange.get('timestamp', 'N/A')}")
            export_lines.append(f"Usuario: {exchange['query']}")
            export_lines.append(f"Chatbot: {exchange['response']}")
            
            if exchange.get('relevant_docs'):
                docs = [doc for doc in exchange['relevant_docs'] if doc]
                if docs:
                    export_lines.append(f"Documentos usados: {', '.join(docs)}")
            
            export_lines.append("")
        
        return "\n".join(export_lines)
    
    def suggest_questions(self, n_suggestions: int = 3) -> List[str]:
        """
        Sugiere preguntas basadas en los documentos disponibles.
        
        Args:
            n_suggestions (int): Número de sugerencias a generar
            
        Returns:
            List[str]: Lista de preguntas sugeridas
        """
        try:
            # Obtener información sobre los documentos disponibles
            db_info = self.indexer.get_collection_info()
            
            if db_info['total_documents'] == 0:
                return ["Primero necesitas indexar algunos documentos."]
            
            # Obtener algunos documentos para generar sugerencias
            sample_docs = self.indexer.search_documents("contenido información", n_results=2)
            
            if not sample_docs:
                return ["No hay suficiente contenido para generar sugerencias."]
            
            # Crear un prompt para generar sugerencias
            context = "\n".join([doc['content'][:200] + "..." for doc in sample_docs])
            
            prompt = f"""Basándote en estos fragmentos de documentos:

{context}

Sugiere {n_suggestions} preguntas interesantes que un usuario podría hacer sobre este contenido. 
Las preguntas deben ser específicas y útiles.
Devuelve solo las preguntas, una por línea, sin numeración."""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.5
            )
            
            suggestions = response.choices[0].message.content.strip().split('\n')
            return [q.strip() for q in suggestions if q.strip()][:n_suggestions]
            
        except Exception as e:
            logger.error(f"Error generando sugerencias: {e}")
            return ["Error generando sugerencias de preguntas."]
    
    def _get_current_timestamp(self) -> str:
        """
        Obtiene el timestamp actual formateado.
        
        Returns:
            str: Timestamp en formato legible
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def set_system_prompt(self, new_prompt: str):
        """
        Permite cambiar el prompt del sistema.
        
        Args:
            new_prompt (str): Nuevo prompt del sistema
        """
        self.system_prompt = new_prompt
        logger.info("Prompt del sistema actualizado")
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Obtiene información de la base de datos.
        
        Returns:
            Dict[str, Any]: Información de la base de datos
        """
        return self.indexer.get_collection_info()
    
    def list_available_documents(self) -> List[str]:
        """
        Lista todos los documentos disponibles en la base de datos.
        
        Returns:
            List[str]: Lista de nombres de documentos
        """
        return self.indexer.list_documents()