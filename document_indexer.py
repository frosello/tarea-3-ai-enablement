"""
DocumentIndexer - Clase para indexar documentos usando embeddings de OpenAI y ChromaDB.

Esta clase se encarga de:
- Dividir documentos en chunks manejables
- Generar embeddings usando la API de OpenAI
- Almacenar y buscar documentos en ChromaDB
- Gestionar la base de datos vectorial
"""

import os
import logging
from typing import List, Dict, Any, Optional

import tiktoken
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# Configurar logging
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = os.getenv("CHROMA_DB_PERSIST_DIR", "./chroma_db")

# Verificar que la API key de OpenAI esté configurada
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY no está configurada en el archivo .env")


class DocumentIndexer:
    """Clase para indexar documentos usando embeddings de OpenAI y ChromaDB."""
    
    def __init__(self, collection_name: str = "documents"):
        """
        Inicializa el DocumentIndexer.
        
        Args:
            collection_name (str): Nombre de la colección en ChromaDB
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Configurar ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Obtener o crear la colección
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Colección '{collection_name}' cargada exitosamente")
        except Exception:
            self.collection = self.chroma_client.create_collection(collection_name)
            logger.info(f"Nueva colección '{collection_name}' creada")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Obtiene el embedding de un texto usando OpenAI.
        
        Args:
            text (str): Texto para generar embedding
            
        Returns:
            List[float]: Vector embedding del texto
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error al obtener embedding: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Cuenta los tokens en un texto.
        
        Args:
            text (str): Texto a analizar
            
        Returns:
            int: Número de tokens
        """
        return len(self.encoding.encode(text))
    
    def split_text_into_chunks(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        """
        Divide un texto en chunks más pequeños.
        
        Args:
            text (str): Texto a dividir
            max_tokens (int): Máximo número de tokens por chunk
            overlap (int): Número de palabras de solapamiento entre chunks
            
        Returns:
            List[str]: Lista de chunks de texto
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.count_tokens(word + " ")
            
            if current_tokens + word_tokens > max_tokens and current_chunk:
                # Guardar el chunk actual
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Comenzar nuevo chunk con solapamiento
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words + [word]
                current_tokens = self.count_tokens(" ".join(current_chunk))
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        
        # Añadir el último chunk si no está vacío
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def index_document(self, content: str, filename: str, metadata: Optional[Dict] = None) -> int:
        """
        Indexa un documento dividiéndolo en chunks y creando embeddings.
        
        Args:
            content (str): Contenido del documento
            filename (str): Nombre del archivo
            metadata (Optional[Dict]): Metadatos adicionales del documento
            
        Returns:
            int: Número de chunks indexados exitosamente
        """
        if not content.strip():
            logger.warning(f"El documento '{filename}' está vacío")
            return 0
        
        # Dividir el contenido en chunks
        chunks = self.split_text_into_chunks(content)
        logger.info(f"Documento '{filename}' dividido en {len(chunks)} chunks")
        
        # Crear embeddings para cada chunk
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        base_metadata = metadata or {}
        base_metadata["filename"] = filename
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_{i}"
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            try:
                embedding = self.get_embedding(chunk)
                
                documents.append(chunk)
                embeddings.append(embedding)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
                
            except Exception as e:
                logger.error(f"Error procesando chunk {i} del documento '{filename}': {e}")
                continue
        
        # Agregar a la colección de ChromaDB
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Indexados {len(documents)} chunks del documento '{filename}'")
                return len(documents)
            except Exception as e:
                logger.error(f"Error agregando chunks a ChromaDB: {e}")
                return 0
        else:
            logger.warning(f"No se pudieron procesar chunks para el documento '{filename}'")
            return 0
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Busca documentos similares a la consulta.
        
        Args:
            query (str): Consulta de búsqueda
            n_results (int): Número de resultados a retornar
            
        Returns:
            List[Dict[str, Any]]: Lista de documentos relevantes con metadatos
        """
        try:
            query_embedding = self.get_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            search_results = []
            for i in range(len(results['documents'][0])):
                search_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error en la búsqueda: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre la colección.
        
        Returns:
            Dict[str, Any]: Información de la colección
        """
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Error obteniendo información de la colección: {e}")
            return {"total_documents": 0, "collection_name": "unknown"}
    
    def delete_collection(self) -> bool:
        """
        Elimina la colección actual.
        
        Returns:
            bool: True si se eliminó exitosamente
        """
        try:
            self.chroma_client.delete_collection(self.collection.name)
            logger.info(f"Colección '{self.collection.name}' eliminada")
            return True
        except Exception as e:
            logger.error(f"Error eliminando colección: {e}")
            return False
    
    def list_documents(self) -> List[str]:
        """
        Lista todos los documentos únicos en la colección.
        
        Returns:
            List[str]: Lista de nombres de archivos
        """
        try:
            # Obtener todos los metadatos
            results = self.collection.get()
            filenames = set()
            
            for metadata in results['metadatas']:
                if 'filename' in metadata:
                    filenames.add(metadata['filename'])
            
            return list(filenames)
            
        except Exception as e:
            logger.error(f"Error listando documentos: {e}")
            return []