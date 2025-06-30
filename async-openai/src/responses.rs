use crate::{
    config::Config,
    error::OpenAIError,
    types::responses::{CreateResponse, Response, ResponseStream},
    Client,
};

/// Given text input or a list of context items, the model will generate a response.
///
/// Related guide: [Responses API](https://platform.openai.com/docs/guides/responses)
pub struct Responses<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Responses<'c, C> {
    /// Constructs a new Responses client.
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    /// Creates a model response for the given input.
    #[crate::byot(
        T0 = serde::Serialize,
        R = serde::de::DeserializeOwned
    )]
    pub async fn create(&self, request: CreateResponse) -> Result<Response, OpenAIError> {
        self.client.post("/responses", request).await
    }

    /// Create a model response for the given input.
    ///
    /// byot: You must ensure "stream: true" in serialized `request`
    #[crate::byot(
        T0 = serde::Serialize,
        R = serde::de::DeserializeOwned,
        stream = "true",
        where_clause = "R: std::marker::Send + 'static + TryFrom<eventsource_stream::Event, Error = OpenAIError>"
    )]
    pub async fn create_stream(
        &self,
        request: CreateResponse,
    ) -> Result<ResponseStream, OpenAIError> {
        #[allow(unused_mut)]
        let mut request = request;
        #[cfg(not(feature = "byot"))]
        {
            if request.stream.is_some() && !request.stream.unwrap() {
                return Err(OpenAIError::InvalidArgument(
                    "When stream is false, use Responses::create".into(),
                ));
            }

            request.stream = Some(true);
        }
        Ok(self
            .client
            .post_stream_mapped_raw_events("/responses", request, TryFrom::try_from)
            .await)
    }
}
