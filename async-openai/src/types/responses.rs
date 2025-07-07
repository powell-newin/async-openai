use crate::error::{map_deserialization_error, ApiError, OpenAIError};
pub use crate::types::{
    CompletionTokensDetails, ImageDetail, PromptTokensDetails, ReasoningEffort,
    ResponseFormatJsonSchema,
};
use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, pin::Pin};

/// Role of messages in the API.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
    Developer,
}

/// Status of input/output items.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OutputStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// Input payload: raw text or structured context items.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Input {
    /// A text input to the model, equivalent to a text input with the user role.
    Text(String),
    /// A list of one or many input items to the model, containing different content types.
    Items(Vec<InputItem>),
}

/// A context item: currently only messages.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged, rename_all = "snake_case")]
pub enum InputItem {
    Message(InputMessage),
    Custom(serde_json::Value),
}

/// A message to prime the model.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "InputMessageArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct InputMessage {
    #[serde(default, rename = "type")]
    pub kind: InputMessageType,
    /// The role of the message input.
    pub role: Role,
    /// Text, image, or audio input to the model, used to generate a response. Can also contain
    /// previous assistant responses.
    pub content: InputContent,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum InputMessageType {
    #[default]
    Message,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum InputContent {
    /// A text input to the model.
    TextInput(String),
    /// A list of one or many input items to the model, containing different content types.
    InputItemContentList(Vec<ContentType>),
}

/// Parts of a message: text, image, file, or audio.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentType {
    /// A text input to the model.
    InputText(InputText),
    /// An image input to the model.
    InputImage(InputImage),
    /// A file input to the model.
    InputFile(InputFile),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "InputTextArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct InputText {
    text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "InputImageArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct InputImage {
    /// The detail level of the image to be sent to the model.
    detail: ImageDetail,
    /// The ID of the file to be sent to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
    /// The URL of the image to be sent to the model. A fully qualified URL or base64 encoded image
    /// in a data URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "InputFileArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct InputFile {
    /// The content of the file to be sent to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    file_data: Option<String>,
    /// The ID of the file to be sent to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
    /// The name of the file to be sent to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    filename: Option<String>,
}

/// Builder for a Responses API request.
#[derive(Clone, Serialize, Deserialize, Debug, Default, Builder, PartialEq)]
#[builder(
    name = "CreateResponseArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CreateResponse {
    /// Text, image, or file inputs to the model, used to generate a response.
    pub input: Input,

    /// Model ID used to generate the response, like `gpt-4o`.
    /// OpenAI offers a wide range of models with different capabilities,
    /// performance characteristics, and price points.
    pub model: String,

    /// Whether to run the model response in the background.
    /// boolean or null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    /// Specify additional output data to include in the model response.
    ///
    /// Supported values:
    /// - `file_search_call.results`
    ///   Include the search results of the file search tool call.
    /// - `message.input_image.image_url`
    ///   Include image URLs from the input message.
    /// - `computer_call_output.output.image_url`
    ///   Include image URLs from the computer call output.
    /// - `reasoning.encrypted_content`
    ///   Include an encrypted version of reasoning tokens in reasoning item outputs.
    ///   This enables reasoning items to be used in multi-turn conversations when
    ///   using the Responses API statelessly (for example, when the `store` parameter
    ///   is set to `false`, or when an organization is enrolled in the zero-data-
    ///   retention program).
    ///
    /// If `None`, no additional data is returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,

    /// Inserts a system (or developer) message as the first item in the model's context.
    ///
    /// When using along with previous_response_id, the instructions from a previous response will
    /// not be carried over to the next response. This makes it simple to swap out system
    /// (or developer) messages in new responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// An upper bound for the number of tokens that can be generated for a
    /// response, including visible output tokens and reasoning tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// The maximum number of total calls to built-in tools that can be processed in a response.
    /// This maximum number applies across all built-in tool calls, not per individual tool.
    /// Any further attempts to call a tool by the model will be ignored.
    pub max_tool_calls: Option<u32>,

    /// Set of 16 key-value pairs that can be attached to an object. This can be
    /// useful for storing additional information about the object in a structured
    /// format, and querying for objects via API or the dashboard.
    ///
    /// Keys are strings with a maximum length of 64 characters. Values are
    /// strings with a maximum length of 512 characters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,

    /// Whether to allow the model to run tool calls in parallel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// The unique ID of the previous response to the model. Use this to create
    /// multi-turn conversations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Reference to a prompt template and its variables.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptConfig>,

    /// **o-series models only**: Configuration options for reasoning models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    /// Specifies the latency tier to use for processing the request.
    ///
    /// This parameter is relevant for customers subscribed to the Scale tier service.
    ///
    /// Supported values:
    /// - `auto`
    ///   - If the Project is Scale tier enabled, the system will utilize Scale tier credits until
    ///     they are exhausted.
    ///   - If the Project is not Scale tier enabled, the request will be processed using the
    ///     default service tier with a lower uptime SLA and no latency guarantee.
    /// - `default`
    ///   The request will be processed using the default service tier with a lower uptime SLA and
    ///   no latency guarantee.
    /// - `flex`
    ///   The request will be processed with the Flex Processing service tier. Learn more.
    ///
    /// When not set, the default behavior is `auto`.
    ///
    /// When this parameter is set, the response body will include the `service_tier` utilized.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,

    /// Whether to store the generated model response for later retrieval via API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// If set to true, the model response data will be streamed to the client as it is
    /// generated using server-sent events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8
    /// will make the output more random, while lower values like 0.2 will make it
    /// more focused and deterministic. We generally recommend altering this or
    /// `top_p` but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Configuration options for a text response from the model. Can be plain text
    /// or structured JSON data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,

    /// How the model should select which tool (or tools) to use when generating
    /// a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// An array of tools the model may call while generating a response.
    /// Can include built-in tools (file_search, web_search_preview,
    /// computer_use_preview) or custom function definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// An integer between 0 and 20 specifying the number of most likely tokens to return
    /// at each token position, each with an associated log probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>, // TODO add validation of range

    /// An alternative to sampling with temperature, called nucleus sampling,
    /// where the model considers the results of the tokens with top_p probability
    /// mass. So 0.1 means only the tokens comprising the top 10% probability mass
    /// are considered. We generally recommend altering this or `temperature` but
    /// not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// The truncation strategy to use for the model response:
    /// - `auto`: drop items in the middle to fit context window.
    /// - `disabled`: error if exceeding context window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,

    /// A unique identifier representing your end-user, which can help OpenAI to
    /// monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Service tier request options.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct PromptConfig {
    /// The unique identifier of the prompt template to use.
    pub id: String,

    /// Optional version of the prompt template.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Optional map of values to substitute in for variables in your prompt. The substitution
    /// values can either be strings, or other Response input types like images or files.
    /// For now only supporting Strings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<HashMap<String, String>>,
}

/// Service tier request options.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
}

/// Truncation strategies.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Truncation {
    Auto,
    Disabled,
}

/// o-series reasoning settings.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "ReasoningConfigArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ReasoningConfig {
    /// Constrain effort on reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Summary mode for reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

/// Configuration for text response format.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct TextConfig {
    /// Defines the format: plain text, JSON object, or JSON schema.
    pub format: TextResponseFormat,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextResponseFormat {
    /// The type of response format being defined: `text`
    Text,
    /// The type of response format being defined: `json_object`
    JsonObject,
    /// The type of response format being defined: `json_schema`
    JsonSchema(ResponseFormatJsonSchema),
}

/// Definitions for model-callable tools.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    /// File search tool.
    FileSearch(FileSearch),
    /// Custom function call.
    Function(Function),
    /// Web search preview tool.
    WebSearchPreview(WebSearchPreview),
    /// Virtual computer control tool.
    ComputerUsePreview(ComputerUsePreview),
    /// Remote Model Context Protocol server.
    Mcp(Mcp),
    /// Python code interpreter tool.
    CodeInterpreter(CodeInterpreter),
    /// Image generation tool.
    ImageGeneration(ImageGeneration),
    /// Local shell command execution tool.
    LocalShell,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "FileSearchArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct FileSearch {
    /// The IDs of the vector stores to search.
    pub vector_store_ids: Vec<String>,
    /// The maximum number of results to return. This number should be between 1 and 50 inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_results: Option<u32>,
    /// A filter to apply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<Filter>,
    /// Ranking options for search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_options: Option<RankingOptions>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "FunctionArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
pub struct Function {
    /// The name of the function to call.
    pub name: String,
    /// A JSON schema object describing the parameters of the function.
    pub parameters: serde_json::Value,
    /// Whether to enforce strict parameter validation.
    pub strict: bool,
    /// A description of the function. Used by the model to determine whether or not to call the
    /// function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "WebSearchPreviewArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
pub struct WebSearchPreview {
    /// The user's location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<Location>,
    /// High level guidance for the amount of context window space to use for the search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<WebSearchContextSize>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchContextSize {
    Low,
    Medium,
    High,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "ComputerUsePreviewArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
pub struct ComputerUsePreview {
    /// The type of computer environment to control.
    environment: String,
    /// The width of the computer display.
    display_width: u32,
    /// The height of the computer display.
    display_height: u32,
}

/// Options for search result ranking.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct RankingOptions {
    /// The ranker to use for the file search.
    pub ranker: String,
    /// The score threshold for the file search, a number between 0 and 1. Numbers closer to 1 will
    /// attempt to return only the most relevant results, but may return fewer results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f32>,
}

/// Filters for file search.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Filter {
    /// A filter used to compare a specified attribute key to a given value using a defined
    /// comparison operation.
    Comparison(ComparisonFilter),
    /// Combine multiple filters using and or or.
    Compound(CompoundFilter),
}

/// Single comparison filter.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ComparisonFilter {
    /// Specifies the comparison operator
    #[serde(rename = "type")]
    pub op: ComparisonType,
    /// The key to compare against the value.
    pub key: String,
    /// The value to compare against the attribute key; supports string, number, or boolean types.
    pub value: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum ComparisonType {
    #[serde(rename = "eq")]
    Equals,
    #[serde(rename = "ne")]
    NotEquals,
    #[serde(rename = "gt")]
    GreaterThan,
    #[serde(rename = "gte")]
    GreaterThanOrEqualTo,
    #[serde(rename = "lt")]
    LessThan,
    #[serde(rename = "lte")]
    LessThanOrEqualTo,
}

/// Combine multiple filters.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CompoundFilter {
    /// Type of operation
    #[serde(rename = "type")]
    pub op: ComparisonType,
    /// Array of filters to combine. Items can be ComparisonFilter or CompoundFilter.
    pub filters: Vec<Filter>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CompoundType {
    And,
    Or,
}

/// Approximate user location for web search.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "LocationArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct Location {
    /// The type of location approximation. Always approximate.
    #[serde(rename = "type")]
    pub kind: String,
    /// Free text input for the city of the user, e.g. San Francisco.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// The two-letter ISO country code of the user, e.g. US.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Free text input for the region of the user, e.g. California.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// The IANA timezone of the user, e.g. America/Los_Angeles.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

/// MCP (Model Context Protocol) tool configuration.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "McpArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct Mcp {
    /// A label for this MCP server.
    pub server_label: String,
    /// The URL for the MCP server.
    pub server_url: String,
    /// List of allowed tool names or filter object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<AllowedTools>,
    /// Optional HTTP headers for the MCP server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<Value>,
    /// Approval policy or filter for tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_approval: Option<RequireApproval>,
}

/// Allowed tools configuration for MCP.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum AllowedTools {
    /// A flat list of allowed tool names.
    List(Vec<String>),
    /// A filter object specifying allowed tools.
    Filter(McpAllowedToolsFilter),
}

/// Filter object for MCP allowed tools.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct McpAllowedToolsFilter {
    /// Names of tools in the filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_names: Option<Vec<String>>,
}

/// Approval policy or filter for MCP tools.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum RequireApproval {
    /// A blanket policy: "always" or "never".
    Policy(RequireApprovalPolicy),
    /// A filter object specifying which tools require approval.
    Filter(McpApprovalFilter),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RequireApprovalPolicy {
    Always,
    Never,
}

/// Filter object for MCP tool approval.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct McpApprovalFilter {
    /// A list of tools that always require approval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub always: Option<McpAllowedToolsFilter>,
    /// A list of tools that never require approval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub never: Option<McpAllowedToolsFilter>,
}

/// Container configuration for a code interpreter.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum CodeInterpreterContainer {
    /// A simple container ID.
    Id(String),
    /// Auto-configured container with optional files.
    Container(CodeInterpreterContainerKind),
}

/// Auto configuration for code interpreter container.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodeInterpreterContainerKind {
    Auto {
        /// Optional list of uploaded file IDs.
        #[serde(skip_serializing_if = "Option::is_none")]
        file_ids: Option<Vec<String>>,
    },
}

/// Code interpreter tool definition.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "CodeInterpreterArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CodeInterpreter {
    /// Container configuration for running code.
    pub container: CodeInterpreterContainer,
}

/// Mask image input for image generation.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputImageMask {
    /// Base64-encoded mask image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    /// File ID for the mask image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
}

/// Image generation tool definition.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(
    name = "ImageGenerationArgs",
    pattern = "mutable",
    setter(into, strip_option),
    default
)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ImageGeneration {
    /// Background type: transparent, opaque, or auto.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<ImageGenerationBackground>,
    /// Optional mask for inpainting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_image_mask: Option<InputImageMask>,
    /// Model to use (default: gpt-image-1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Moderation level (default: auto).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moderation: Option<String>,
    /// Compression level (0-100).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<u8>,
    /// Output format: png, webp, or jpeg.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<ImageGenerationOutputFormat>,
    /// Number of partial images (0-3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_images: Option<u8>,
    /// Quality: low, medium, high, or auto.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<ImageGenerationQuality>,
    /// Size: e.g. "1024x1024" or auto.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<ImageGenerationSize>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageGenerationBackground {
    Transparent,
    Opaque,
    Auto,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageGenerationOutputFormat {
    Png,
    Webp,
    Jpeg,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageGenerationQuality {
    Low,
    Medium,
    High,
    Auto,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageGenerationSize {
    Auto,
    #[serde(rename = "1024x1024")]
    Size1024x1024,
    #[serde(rename = "1024x1536")]
    Size1024x1536,
    #[serde(rename = "1536x1024")]
    Size1536x1024,
}

/// Control how the model picks or is forced to pick a tool.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Controls which (if any) tool is called by the model.
    Mode(ToolChoiceMode),
    /// Indicates that the model should use a built-in tool to generate a response.
    Hosted {
        /// The type of hosted tool the model should to use.
        #[serde(rename = "type")]
        kind: HostedToolType,
    },
    /// Use this option to force the model to call a specific function.
    Function {
        /// The name of the function to call.
        name: String,
    },
}

/// Simple tool-choice modes.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    /// The model will not call any tool and instead generates a message.
    None,
    /// The model can pick between generating a message or calling one or more tools.
    Auto,
    /// The model must call one or more tools.
    Required,
}

/// Hosted tool type identifiers.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum HostedToolType {
    FileSearch,
    WebSearchPreview,
    ComputerUsePreview,
}

/// Error returned by the API when a request fails.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ErrorObject {
    /// The error code for the response.
    pub code: String,
    /// A human-readable description of the error.
    pub message: String,
}

/// Details about an incomplete response.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct IncompleteDetails {
    /// The reason why the response is incomplete.
    pub reason: String,
}

/// A simple text output from the model.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct OutputText {
    /// The annotations of the text output.
    pub annotations: Vec<Annotation>,
    /// The text output from the model.
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    /// A citation to a file.
    FileCitation(FileCitation),
    /// A citation for a web resource used to generate a model response.
    UrlCitation(UrlCitation),
    /// A path to a file.
    FilePath(FilePath),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FileCitation {
    /// The ID of the file.
    file_id: String,
    /// The index of the file in the list of files.
    index: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct UrlCitation {
    /// The index of the last character of the URL citation in the message.
    end_index: u32,
    /// The index of the first character of the URL citation in the message.
    start_index: u32,
    /// The title of the web resource.
    title: String,
    /// The URL of the web resource.
    url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FilePath {
    /// The ID of the file.
    file_id: String,
    /// The index of the file in the list of files.
    index: u32,
}

/// A refusal explanation from the model.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Refusal {
    /// The refusal explanationfrom the model.
    pub refusal: String,
}

/// A message generated by the model.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct OutputMessage {
    /// The content of the output message.
    pub content: Vec<Content>,
    /// The unique ID of the output message.
    pub id: String,
    /// The role of the output message. Always assistant.
    pub role: Role,
    /// The status of the message input.
    pub status: OutputStatus,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// A text output from the model.
    OutputText(OutputText),
    /// A refusal from the model.
    Refusal(Refusal),
}

/// Nested content within an output message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContent {
    /// An output message from the model.
    Message(OutputMessage),
    /// The results of a file search tool call.
    FileSearchCall(FileSearchCallOutput),
    /// A tool call to run a function.
    FunctionCall(FunctionCall),
    /// The results of a web search tool call.
    WebSearchCall(WebSearchCallOutput),
    /// A tool call to a computer use tool.
    ComputerCall(ComputerCallOutput),
    /// A description of the chain of thought used by a reasoning model while generating a response.
    /// Be sure to include these items in your input to the Responses API for subsequent turns of a
    /// conversation if you are manually managing context.
    Reasoning(ReasoningItem),
    /// Image generation tool call output.
    ImageGenerationCall(ImageGenerationCallOutput),
    /// Code interpreter tool call output.
    CodeInterpreterCall(CodeInterpreterCallOutput),
    /// Local shell tool call output.
    LocalShellCall(LocalShellCallOutput),
    /// MCP tool invocation output.
    McpCall(McpCallOutput),
    /// MCP list-tools output.
    McpListTools(McpListToolsOutput),
    /// MCP approval request output.
    McpApprovalRequest(McpApprovalRequestOutput),
}

/// A reasoning item representing the model's chain of thought, including summary paragraphs.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ReasoningItem {
    /// Unique identifier of the reasoning content.
    pub id: String,
    /// The summarized chain-of-thought paragraphs.
    pub summary: Vec<SummaryText>,
    /// The encrypted content of the reasoning item - populated when a response is generated with
    /// `reasoning.encrypted_content` in the `include` parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    /// The status of the reasoning item.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<OutputStatus>,
}

/// A single summary text fragment from reasoning.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SummaryText {
    /// A short summary of the reasoning used by the model.
    pub text: String,
}

/// File search tool call output.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FileSearchCallOutput {
    /// The unique ID of the file search tool call.
    pub id: String,
    /// The queries used to search for files.
    pub queries: Vec<String>,
    /// The status of the file search tool call.
    pub status: FileSearchCallOutputStatus,
    /// The results of the file search tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<FileSearchResult>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FileSearchCallOutputStatus {
    InProgress,
    Searching,
    Incomplete,
    Failed,
    Completed,
}

/// A single result from a file search.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FileSearchResult {
    /// The unique ID of the file.
    pub file_id: String,
    /// The name of the file.
    pub filename: String,
    /// The relevance score of the file - a value between 0 and 1.
    pub score: f32,
    /// The text that was retrieved from the file.
    pub text: String,
    /// Set of 16 key-value pairs that can be attached to an object. This can be useful for storing
    /// additional information about the object in a structured format, and querying for objects
    /// API or the dashboard. Keys are strings with a maximum length of 64 characters
    /// . Values are strings with a maximum length of 512 characters, booleans, or numbers.
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SafetyCheck {
    /// The ID of the safety check.
    pub id: String,
    /// The type/code of the pending safety check.
    pub code: String,
    /// Details about the pending safety check.
    pub message: String,
}

/// Web search tool call output.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct WebSearchCallOutput {
    /// The unique ID of the web search tool call.
    pub id: String,
    /// The status of the web search tool call.
    pub status: String,
}

/// Output from a computer tool call.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ComputerCallOutput {
    pub action: ComputerCallAction,
    /// An identifier used when responding to the tool call with output.
    pub call_id: String,
    /// The unique ID of the computer call.
    pub id: String,
    /// The pending safety checks for the computer call.
    pub pending_safety_checks: Vec<SafetyCheck>,
    /// The status of the item.
    pub status: OutputStatus,
}

/// A point in 2D space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

/// Represents all user‐triggered actions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ComputerCallAction {
    /// A click action.
    Click(Click),

    /// A double-click action.
    DoubleClick(DoubleClick),

    /// A drag action.
    Drag(Drag),

    /// A keypress action.
    KeyPress(KeyPress),

    /// A mouse move action.
    Move(MoveAction),

    /// A screenshot action.
    Screenshot,

    /// A scroll action.
    Scroll(Scroll),

    /// A type (text entry) action.
    Type(TypeAction),

    /// A wait (no-op) action.
    Wait,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ButtonPress {
    Left,
    Right,
    Wheel,
    Back,
    Forward,
}

/// A click action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Click {
    /// Which mouse button was pressed.
    pub button: ButtonPress,
    /// X‐coordinate of the click.
    pub x: i32,
    /// Y‐coordinate of the click.
    pub y: i32,
}

/// A double click action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoubleClick {
    /// X‐coordinate of the double click.
    pub x: i32,
    /// Y‐coordinate of the double click.
    pub y: i32,
}

/// A drag action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Drag {
    /// The path of points the cursor drags through.
    pub path: Vec<Point>,
    /// X‐coordinate at the end of the drag.
    pub x: i32,
    /// Y‐coordinate at the end of the drag.
    pub y: i32,
}

/// A keypress action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyPress {
    /// The list of keys to press (e.g. `["Control", "C"]`).
    pub keys: Vec<String>,
}

/// A mouse move action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoveAction {
    /// X‐coordinate to move to.
    pub x: i32,
    /// Y‐coordinate to move to.
    pub y: i32,
}

/// A scroll action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scroll {
    /// Horizontal scroll distance.
    pub scroll_x: i32,
    /// Vertical scroll distance.
    pub scroll_y: i32,
    /// X‐coordinate where the scroll began.
    pub x: i32,
    /// Y‐coordinate where the scroll began.
    pub y: i32,
}

/// A typing (text entry) action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeAction {
    /// The text to type.
    pub text: String,
}

/// Metadata for a function call request.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FunctionCall {
    /// The unique ID of the function tool call.
    pub id: String,
    /// The unique ID of the function tool call generated by the model.
    pub call_id: String,
    /// The name of the function to run.
    pub name: String,
    /// A JSON string of the arguments to pass to the function.
    pub arguments: String,
    /// The status of the item.
    pub status: OutputStatus,
}

/// Output of an image generation request.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ImageGenerationCallOutput {
    /// Unique ID of the image generation call.
    pub id: String,
    /// Base64-encoded generated image, or null.
    pub result: Option<String>,
    /// Status of the image generation call.
    pub status: String,
}

/// Output of a code interpreter request.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CodeInterpreterCallOutput {
    /// The code that was executed.
    pub code: String,
    /// Unique ID of the call.
    pub id: String,
    /// Status of the tool call.
    pub status: String,
    /// ID of the container used to run the code.
    pub container_id: String,
    /// The results of the execution: logs or files.
    pub results: Vec<CodeInterpreterResult>,
}

/// Individual result from a code interpreter: either logs or files.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodeInterpreterResult {
    /// Text logs from the execution.
    Logs(CodeInterpreterTextOutput),
    /// File outputs from the execution.
    Files(CodeInterpreterFileOutput),
}

/// The output containing execution logs.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CodeInterpreterTextOutput {
    /// The logs of the code interpreter tool call.
    pub logs: String,
}

/// The output containing file references.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CodeInterpreterFileOutput {
    /// List of file IDs produced.
    pub files: Vec<CodeInterpreterFile>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CodeInterpreterFile {
    /// The ID of the file.
    file_id: String,
    /// The MIME type of the file.
    mime_type: String,
}

/// Output of a local shell command request.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct LocalShellCallOutput {
    /// Details of the exec action.
    pub action: LocalShellAction,
    /// Unique call identifier for responding to the tool call.
    pub call_id: String,
    /// Unique ID of the local shell call.
    pub id: String,
    /// Status of the local shell call.
    pub status: String,
}

/// Define the shape of a local shell action (exec).
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct LocalShellAction {
    /// The command to run.
    pub command: Vec<String>,
    /// Environment variables to set for the command.
    pub env: HashMap<String, String>,
    /// Optional timeout for the command (ms).
    pub timeout_ms: Option<u64>,
    /// Optional user to run the command as.
    pub user: Option<String>,
    /// Optional working directory for the command.
    pub working_directory: Option<String>,
}

/// Output of an MCP server tool invocation.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct McpCallOutput {
    /// JSON string of the arguments passed.
    pub arguments: String,
    /// Unique ID of the MCP call.
    pub id: String,
    /// Name of the tool invoked.
    pub name: String,
    /// Label of the MCP server.
    pub server_label: String,
    /// Error message from the call, if any.
    pub error: Option<String>,
    /// Output from the call, if any.
    pub output: Option<String>,
}

/// Output listing tools available on an MCP server.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct McpListToolsOutput {
    /// Unique ID of the list request.
    pub id: String,
    /// Label of the MCP server.
    pub server_label: String,
    /// Tools available on the server with metadata.
    pub tools: Vec<McpToolInfo>,
    /// Error message if listing failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Information about a single tool on an MCP server.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct McpToolInfo {
    /// The name of the tool.
    pub name: String,
    /// The JSON schema describing the tool's input.
    pub input_schema: Value,
    /// Additional annotations about the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// The description of the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Output representing a human approval request for an MCP tool.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct McpApprovalRequestOutput {
    /// JSON string of arguments for the tool.
    pub arguments: String,
    /// Unique ID of the approval request.
    pub id: String,
    /// Name of the tool requiring approval.
    pub name: String,
    /// Label of the MCP server making the request.
    pub server_label: String,
}

/// Usage statistics for a response.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Usage {
    /// The number of input tokens.
    pub input_tokens: u32,
    /// A detailed breakdown of the input tokens.
    pub input_tokens_details: PromptTokensDetails,
    /// The number of output tokens.
    pub output_tokens: u32,
    /// A detailed breakdown of the output tokens.
    pub output_tokens_details: CompletionTokensDetails,
    /// The total number of tokens used.
    pub total_tokens: u32,
}

/// The complete response returned by the Responses API.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Response {
    /// Unix timestamp (in seconds) when this Response was created.
    pub created_at: u64,

    /// Error object if the API failed to generate a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorObject>,

    /// Unique identifier for this response.
    pub id: String,

    /// Details about why the response is incomplete, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,

    /// Instructions that were inserted as the first item in context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// The value of `max_output_tokens` that was honored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Metadata tags/values that were attached to this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,

    /// Model ID used to generate the response.
    pub model: String,

    /// The object type – always `response`.
    pub object: String,

    /// The array of content items generated by the model.
    pub output: Vec<OutputContent>,

    /// SDK-only convenience property that contains the aggregated text output from all
    /// `output_text` items in the `output` array, if any are present.
    /// Supported in the Python and JavaScript SDKs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_text: Option<String>,

    /// Whether parallel tool calls were enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Previous response ID, if creating part of a multi-turn conversation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Reasoning configuration echoed back (effort, summary settings).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    /// Whether to store the generated model response for later retrieval via API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// The service tier that actually processed this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,

    /// The status of the response generation.
    pub status: Status,

    /// Sampling temperature that was used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Text format configuration echoed back (plain, json_object, json_schema).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,

    /// How the model chose or was forced to choose a tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Tool definitions that were provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// Nucleus sampling cutoff that was used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Truncation strategy that was applied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,

    /// Token usage statistics for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// End-user ID for which this response was generated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Completed,
    Failed,
    InProgress,
    Incomplete,
}

pub type ResponseStream =
    Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent, OpenAIError>> + Send>>;

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "event", content = "data")]
#[non_exhaustive]
#[allow(clippy::large_enum_variant)]
pub enum ResponseStreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated(ResponseCreated),
    #[serde(rename = "response.in_progress")]
    ResponseInProgress(ResponseInProgress),
    #[serde(rename = "response.completed")]
    ResponseCompleted(ResponseCompleted),
    #[serde(rename = "response.failed")]
    ResponseFailed(ResponseFailed),
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete(ResponseIncomplete),
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded(ResponseOutputItemAdded),
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone(ResponseOutputItemDone),
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded(ResponseContentPartAdded),
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone(ResponseContentPartDone),
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta(ResponseOutputTextDelta),
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone(ResponseOutputTextDone),
    #[serde(rename = "response.refusal.delta")]
    ResponseRefusalDelta(ResponseRefusalDelta),
    #[serde(rename = "response.refusal.done")]
    ResponseRefusalDone(ResponseRefusalDone),
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta(ResponseFunctionCallArgumentsDelta),
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone(ResponseFunctionCallArgumentsDone),
    #[serde(rename = "response.file_search_call.in_progress")]
    ResponseFileSearchCallInProgress(ResponseFileSearchCallInProgress),
    #[serde(rename = "response.file_search_call.searching")]
    ResponseFileSearchCallSearching(ResponseFileSearchCallSearching),
    #[serde(rename = "response.file_search_call.completed")]
    ResponseFileSearchCallCompleted(ResponseFileSearchCallCompleted),
    #[serde(rename = "response.web_search_call.in_progress")]
    ResponseWebSearchCallInProgress(ResponseWebSearchCallInProgress),
    #[serde(rename = "response.web_search_call.searching")]
    ResponseWebSearchCallSearching(ResponseWebSearchCallSearching),
    #[serde(rename = "response.web_search_call.completed")]
    ResponseWebSearchCallCompleted(ResponseWebSearchCallCompleted),
    #[serde(rename = "response.reasoning_summary_part.added")]
    ResponseReasoningSummaryPartAdded(ResponseReasoningSummaryPartAdded),
    #[serde(rename = "response.reasoning_summary_part.done")]
    ResponseReasoningSummaryPartDone(ResponseReasoningSummaryPartDone),
    #[serde(rename = "response.image_generation_call.completed")]
    ResponseImageGenerationCallCompleted(ResponseImageGenerationCallCompleted),
    #[serde(rename = "response.image_generation_call.generating")]
    ResponseImageGenerationCallGenerating(ResponseImageGenerationCallGenerating),
    #[serde(rename = "response.image_generation_call.in_progress")]
    ResponseImageGenerationCallInProgress(ResponseImageGenerationCallInProgress),
    #[serde(rename = "response.image_generation_call.partial_image")]
    ResponseImageGenerationCallPartialImage(ResponseImageGenerationCallPartialImage),
    #[serde(rename = "response.mcp_call_arguments.delta")]
    ResponseMcpCallArgumentsDelta(ResponseMcpCallArgumentsDelta),
    #[serde(rename = "response.mcp_call_arguments.done")]
    ResponseMcpCallArgumentsDone(ResponseMcpCallArgumentsDone),
    #[serde(rename = "response.mcp_call.completed")]
    ResponseMcpCallCompleted(ResponseMcpCallCompleted),
    #[serde(rename = "response.mcp_call.failed")]
    ResponseMcpCallFailed(ResponseMcpCallFailed),
    #[serde(rename = "response.mcp_call.in_progress")]
    ResponseMcpCallInProgress(ResponseMcpCallInProgress),
    #[serde(rename = "response.mcp_list_tools.completed")]
    ResponseMcpListToolsCompleted(ResponseMcpListToolsCompleted),
    #[serde(rename = "response.mcp_list_tools.failed")]
    ResponseMcpListToolsFailed(ResponseMcpListToolsFailed),
    #[serde(rename = "response.mcp_list_tools.in_progress")]
    ResponseMcpListToolsInProgress(ResponseMcpListToolsInProgress),
    #[serde(rename = "response.code_interpreter_call.in_progress")]
    ResponseCodeInterpreterCallInProgress(ResponseCodeInterpreterCallInProgress),
    #[serde(rename = "response.code_interpreter_call.interpreting")]
    ResponseCodeInterpreterCallInterpreting(ResponseCodeInterpreterCallInterpreting),
    #[serde(rename = "response.code_interpreter_call.completed")]
    ResponseCodeInterpreterCallCompleted(ResponseCodeInterpreterCallCompleted),
    #[serde(rename = "response.code_interpreter_call_code.delta")]
    ResponseCodeInterpreterCallCodeDelta(ResponseCodeInterpreterCallCodeDelta),
    #[serde(rename = "response.code_interpreter_call_code.done")]
    ResponseCodeInterpreterCallCodeDone(ResponseCodeInterpreterCallCodeDone),
    #[serde(rename = "response.output_text.annotation.added")]
    ResponseOutputTextAnnotationAdded(ResponseOutputTextAnnotationAdded),
    #[serde(rename = "response.queued")]
    ResponseQueued(ResponseQueued),
    #[serde(rename = "response.reasoning.delta")]
    ResponseReasoningDelta(ResponseReasoningDelta),
    #[serde(rename = "response.reasoning.done")]
    ResponseReasoningDone(ResponseReasoningDone),
    #[serde(rename = "response.reasoning_summary.delta")]
    ResponseReasoningSummaryDelta(ResponseReasoningSummaryDelta),
    #[serde(rename = "response.reasoning_summary.done")]
    ResponseReasoningSummaryDone(ResponseReasoningSummaryDone),
    #[serde(rename = "error")]
    ResponseError(ResponseError),
}

impl TryFrom<eventsource_stream::Event> for ResponseStreamEvent {
    type Error = OpenAIError;
    fn try_from(value: eventsource_stream::Event) -> Result<Self, Self::Error> {
        match value.event.as_str() {
            "response.created" => serde_json::from_str::<ResponseCreated>(value.data.as_str())
                .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                .map(ResponseStreamEvent::ResponseCreated),

            "response.in_progress" => {
                serde_json::from_str::<ResponseInProgress>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseInProgress)
            }

            "response.completed" => serde_json::from_str::<ResponseCompleted>(value.data.as_str())
                .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                .map(ResponseStreamEvent::ResponseCompleted),

            "response.failed" => serde_json::from_str::<ResponseFailed>(value.data.as_str())
                .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                .map(ResponseStreamEvent::ResponseFailed),

            "response.incomplete" => {
                serde_json::from_str::<ResponseIncomplete>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseIncomplete)
            }

            "response.output_item.added" => {
                serde_json::from_str::<ResponseOutputItemAdded>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseOutputItemAdded)
            }

            "response.output_item.done" => {
                serde_json::from_str::<ResponseOutputItemDone>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseOutputItemDone)
            }

            "response.content_part.added" => {
                serde_json::from_str::<ResponseContentPartAdded>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseContentPartAdded)
            }

            "response.content_part.done" => {
                serde_json::from_str::<ResponseContentPartDone>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseContentPartDone)
            }

            "response.output_text.delta" => {
                serde_json::from_str::<ResponseOutputTextDelta>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseOutputTextDelta)
            }

            "response.output_text.done" => {
                serde_json::from_str::<ResponseOutputTextDone>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseOutputTextDone)
            }

            "response.refusal.delta" => {
                serde_json::from_str::<ResponseRefusalDelta>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseRefusalDelta)
            }

            "response.refusal.done" => {
                serde_json::from_str::<ResponseRefusalDone>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseRefusalDone)
            }

            "response.function_call_arguments.delta" => {
                serde_json::from_str::<ResponseFunctionCallArgumentsDelta>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseFunctionCallArgumentsDelta)
            }

            "response.function_call_arguments.done" => {
                serde_json::from_str::<ResponseFunctionCallArgumentsDone>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseFunctionCallArgumentsDone)
            }

            "response.file_search_call.in_progress" => {
                serde_json::from_str::<ResponseFileSearchCallInProgress>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseFileSearchCallInProgress)
            }

            "response.file_search_call.searching" => {
                serde_json::from_str::<ResponseFileSearchCallSearching>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseFileSearchCallSearching)
            }

            "response.file_search_call.completed" => {
                serde_json::from_str::<ResponseFileSearchCallCompleted>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseFileSearchCallCompleted)
            }

            "response.web_search_call.in_progress" => {
                serde_json::from_str::<ResponseWebSearchCallInProgress>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseWebSearchCallInProgress)
            }

            "response.web_search_call.searching" => {
                serde_json::from_str::<ResponseWebSearchCallSearching>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseWebSearchCallSearching)
            }

            "response.web_search_call.completed" => {
                serde_json::from_str::<ResponseWebSearchCallCompleted>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseWebSearchCallCompleted)
            }

            "response.reasoning_summary_part.added" => {
                serde_json::from_str::<ResponseReasoningSummaryPartAdded>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseReasoningSummaryPartAdded)
            }

            "response.reasoning_summary_part.done" => {
                serde_json::from_str::<ResponseReasoningSummaryPartDone>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseReasoningSummaryPartDone)
            }

            "response.image_generation_call.completed" => {
                serde_json::from_str::<ResponseImageGenerationCallCompleted>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseImageGenerationCallCompleted)
            }

            "response.image_generation_call.generating" => {
                serde_json::from_str::<ResponseImageGenerationCallGenerating>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseImageGenerationCallGenerating)
            }

            "response.image_generation_call.in_progress" => {
                serde_json::from_str::<ResponseImageGenerationCallInProgress>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseImageGenerationCallInProgress)
            }

            "response.image_generation_call.partial_image" => {
                serde_json::from_str::<ResponseImageGenerationCallPartialImage>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseImageGenerationCallPartialImage)
            }

            "response.mcp_call_arguments.delta" => {
                serde_json::from_str::<ResponseMcpCallArgumentsDelta>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseMcpCallArgumentsDelta)
            }

            "response.mcp_call_arguments.done" => {
                serde_json::from_str::<ResponseMcpCallArgumentsDone>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseMcpCallArgumentsDone)
            }

            "response.mcp_call.completed" => {
                serde_json::from_str::<ResponseMcpCallCompleted>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseMcpCallCompleted)
            }

            "response.mcp_call.failed" => {
                serde_json::from_str::<ResponseMcpCallFailed>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseMcpCallFailed)
            }

            "response.mcp_call.in_progress" => {
                serde_json::from_str::<ResponseMcpCallInProgress>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseMcpCallInProgress)
            }

            "response.mcp_list_tools.completed" => {
                serde_json::from_str::<ResponseMcpListToolsCompleted>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseMcpListToolsCompleted)
            }

            "response.mcp_list_tools.failed" => {
                serde_json::from_str::<ResponseMcpListToolsFailed>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseMcpListToolsFailed)
            }

            "response.mcp_list_tools.in_progress" => {
                serde_json::from_str::<ResponseMcpListToolsInProgress>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseMcpListToolsInProgress)
            }

            "response.code_interpreter_call.in_progress" => {
                serde_json::from_str::<ResponseCodeInterpreterCallInProgress>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseCodeInterpreterCallInProgress)
            }

            "response.code_interpreter_call.interpreting" => {
                serde_json::from_str::<ResponseCodeInterpreterCallInterpreting>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseCodeInterpreterCallInterpreting)
            }

            "response.code_interpreter_call.completed" => {
                serde_json::from_str::<ResponseCodeInterpreterCallCompleted>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseCodeInterpreterCallCompleted)
            }

            "response.code_interpreter_call_code.delta" => {
                serde_json::from_str::<ResponseCodeInterpreterCallCodeDelta>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseCodeInterpreterCallCodeDelta)
            }

            "response.code_interpreter_call_code.done" => {
                serde_json::from_str::<ResponseCodeInterpreterCallCodeDone>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseCodeInterpreterCallCodeDone)
            }

            "response.output_text.annotation.added" => {
                serde_json::from_str::<ResponseOutputTextAnnotationAdded>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseOutputTextAnnotationAdded)
            }

            "response.queued" => serde_json::from_str::<ResponseQueued>(value.data.as_str())
                .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                .map(ResponseStreamEvent::ResponseQueued),

            "response.reasoning.delta" => {
                serde_json::from_str::<ResponseReasoningDelta>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseReasoningDelta)
            }

            "response.reasoning.done" => {
                serde_json::from_str::<ResponseReasoningDone>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseReasoningDone)
            }

            "response.reasoning_summary.delta" => {
                serde_json::from_str::<ResponseReasoningSummaryDelta>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseReasoningSummaryDelta)
            }

            "response.reasoning_summary.done" => {
                serde_json::from_str::<ResponseReasoningSummaryDone>(value.data.as_str())
                    .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                    .map(ResponseStreamEvent::ResponseReasoningSummaryDone)
            }

            "error" => serde_json::from_str::<ResponseError>(value.data.as_str())
                .map_err(|e| map_deserialization_error(e, value.data.as_bytes()))
                .map(ResponseStreamEvent::ResponseError),

            _ => Err(OpenAIError::StreamError(format!(
                "Unrecognized event: {value:?}"
            ))),
        }
    }
}

/// An event that is emitted when a response is created.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseCreated {
    /// The response that was created.
    pub response: Response,
    /// The type of the event. Always `response.created`.
    pub r#type: String,
    /// The sequence number for this event.
    pub sequence_number: i32,
}

/// Emitted when the response is in progress.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseInProgress {
    /// The response that is in progress.
    pub response: Response,
    /// The sequence number for this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.in_progress`.
    pub r#type: String,
}

/// Emitted when the model response is complete.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseCompleted {
    /// Properties of the completed response.
    pub response: Response,
    /// The sequence number for this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.completed`.
    pub r#type: String,
}

/// An event that is emitted when a response fails.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseFailed {
    /// The response that failed.
    pub response: Response,
    /// The sequence number for this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.failed`.
    pub r#type: String,
}

/// An event that is emitted when a response finishes as incomplete.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseIncomplete {
    /// The response that failed.
    pub response: Response,
    /// The sequence number for this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.incomplete`.
    pub r#type: String,
}

/// Emitted when a new output item is added.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseOutputItemAdded {
    /// The output item that was added.
    pub item: OutputContent,
    /// The index of the output item that was added.
    pub output_index: i32,
    /// The sequence number for this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.output_item.added`.
    pub r#type: String,
}

/// Emitted when an output item is marked done.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseOutputItemDone {
    /// The output item that was marked done.
    pub item: OutputContent,
    /// The index of the output item that was marked done.
    pub output_index: i32,
    /// The sequence number for this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.output_item.done`.
    pub r#type: String,
}

/// Emitted when a new content part is added.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseContentPartAdded {
    /// The index of the content part that was added.
    pub content_index: i32,
    /// The ID of the output item that the content part was added to.
    pub item_id: String,
    /// The index of the output item that the content part was added to.
    pub output_index: i32,
    /// The content part that was added.
    pub part: Content,
    /// The sequence number for this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.content_part.added`.
    pub r#type: String,
}

/// Emitted when a content part is done.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseContentPartDone {
    /// The index of the content part that is done.
    pub content_index: i32,
    /// The ID of the output item that the content part was added to.
    pub item_id: String,
    /// The index of the output item that the content part was added to.
    pub output_index: i32,
    /// The content part that is done.
    pub part: Content,
    /// The sequence number for this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.content_part.done`.
    pub r#type: String,
}

/// Emitted when there is an additional text delta.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseOutputTextDelta {
    /// The index of the content part that the text delta was added to.
    pub content_index: i32,
    /// The text delta that was added.
    pub delta: String,
    /// The ID of the output item that the text delta was added to.
    pub item_id: String,
    /// The index of the output item that the content part was added to.
    pub output_index: i32,
    /// The sequence number for this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.output_text.delta`.
    pub r#type: String,
}

/// Emitted when text content is finalized.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseOutputTextDone {
    /// The index of the content part that the text content is finalized.
    pub content_index: i32,
    /// The ID of the output item that the text content is finalized.
    pub item_id: String,
    /// The index of the output item that the text content is finalized.
    pub output_index: i32,
    /// The sequence number for this event.
    pub sequence_number: i32,
    /// The text content that is finalized.
    pub text: String,
    /// The type of the event. Always `response.output_text.delta`.
    pub r#type: String,
}

/// Emitted when there is a partial refusal text.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseRefusalDelta {
    /// The index of the content part that the refusal text is added to.
    pub content_index: i32,
    /// The refusal text that is added.
    pub delta: String,
    /// The ID of the output item that the refusal text is added to.
    pub item_id: String,
    /// The index of the output item that the refusal text is added to.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.refusal.delta`.
    pub r#type: String,
}

/// Emitted when refusal text is finalized.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseRefusalDone {
    /// The index of the content part that the refusal text is finalized.
    pub content_index: i32,
    /// The ID of the output item that the refusal text is finalized.
    pub item_id: String,
    /// The index of the output item that the refusal text is finalized.
    pub output_index: i32,
    /// The refusal text that is finalized.
    pub refusal: String,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.refusal.done`.
    pub r#type: String,
}

/// Emitted when there is a partial function-call arguments delta.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseFunctionCallArgumentsDelta {
    /// The function-call arguments delta that is added.
    pub delta: String,
    /// The ID of the output item that the function-call arguments delta is added to.
    pub item_id: String,
    /// The index of the output item that the function-call arguments delta is added to.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.function_call_arguments.delta`.
    pub r#type: String,
}

/// Emitted when function-call arguments are finalized.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseFunctionCallArgumentsDone {
    /// The function-call arguments.
    pub arguments: String,
    /// The ID of the item.
    pub item_id: String,
    /// The index of the output item.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.function_call_arguments.done`.
    pub r#type: String,
}

/// Emitted when a file search call is initiated.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseFileSearchCallInProgress {
    /// The ID of the output item that the file search call is initiated.
    pub item_id: String,
    /// The index of the output item that the file search call is initiated.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.file_search_call.in_progress`.
    pub r#type: String,
}

/// Emitted when a file search is currently searching.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseFileSearchCallSearching {
    /// The ID of the output item that the file search call is initiated.
    pub item_id: String,
    /// The index of the output item that the file search call is initiated.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.file_search_call.searching`.
    pub r#type: String,
}

/// Emitted when a file search call is completed (results found).
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseFileSearchCallCompleted {
    /// The ID of the output item that the file search call is initiated.
    pub item_id: String,
    /// The index of the output item that the file search call is initiated.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.file_search_call.completed`.
    pub r#type: String,
}

/// Emitted when a web search call is initiated.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseWebSearchCallInProgress {
    /// The ID of the output item that the web search call is initiated.
    pub item_id: String,
    ///The index of the output item that the web search call is associated with.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.web_search_call.in_progress`.
    pub r#type: String,
}

/// Emitted when a web search is currently searching.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseWebSearchCallSearching {
    /// The ID of the output item that the web search call is initiated.
    pub item_id: String,
    /// The index of the output item that the web search call is associated with.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.web_search_call.searching`.
    pub r#type: String,
}

/// Emitted when a web search call is completed.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseWebSearchCallCompleted {
    /// The ID of the output item that the web search call is initiated.
    pub item_id: String,
    /// The index of the output item that the web search call is associated with.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.web_search_call.completed`.
    pub r#type: String,
}

/// Emitted when a new reasoning summary part is added.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseReasoningSummaryPartAdded {
    /// The ID of the item this summary part is associated with.
    pub item_id: String,
    /// The index of the output item this summary part is associated with.
    pub output_index: i32,
    /// The summary part that was added.
    pub part: SummaryPart,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.reasoning_summary_part.added`.
    pub r#type: String,
}

/// Emitted when a reasoning summary part is completed.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseReasoningSummaryPartDone {
    /// The ID of the item this summary part is associated with.
    pub item_id: String,
    /// The index of the output item this summary part is associated with.
    pub output_index: i32,
    /// The completed summary part.
    pub part: SummaryPart,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.reasoning_summary_part.done`.
    pub r#type: String,
}

/// A summary part.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SummaryPart {
    /// The text of the summary part.
    pub text: String,
    /// The type of the summary part. Always `summary_text`.
    pub r#type: String,
}

/// Emitted when an image generation tool call has completed and the final image is available.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseImageGenerationCallCompleted {
    /// The unique identifier of the image generation item being processed.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.image_generation_call.completed`.
    pub r#type: String,
}

/// Emitted when an image generation tool call is actively generating an image (intermediate state).
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseImageGenerationCallGenerating {
    /// The unique identifier of the image generation item being processed.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.image_generation_call.generating`.
    pub r#type: String,
}

/// Emitted when an image generation tool call is in progress.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseImageGenerationCallInProgress {
    /// The unique identifier of the image generation item being processed.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.image_generation_call.in_progress`.
    pub r#type: String,
}

/// Emitted when an image generation tool call is in progress.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseImageGenerationCallPartialImage {
    /// The unique identifier of the image generation item being processed.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// Base64-encoded partial image data, suitable for rendering as an image.
    pub partial_image_b64: String,
    /// 0-based index for the partial image (backend is 1-based, but this is 0-based for the user).
    pub partial_image_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.image_generation_call.partial_image`.
    pub r#type: String,
}

/// Emitted when there is a delta (partial update) to the arguments of an MCP tool call.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseMcpCallArgumentsDelta {
    /// The partial update to the arguments for the MCP tool call.
    pub delta: Value,
    /// The unique identifier of the MCP tool call item being processed.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.mcp_call_arguments_delta'`.
    pub r#type: String,
}

/// Emitted when the arguments for an MCP tool call are finalized.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseMcpCallArgumentsDone {
    /// The finalized arguments for the MCP tool call.
    pub arguments: Value,
    /// The unique identifier of the MCP tool call item being processed.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.mcp_call_arguments.done'`.
    pub r#type: String,
}

/// Emitted when an MCP tool call has completed successfully.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseMcpCallCompleted {
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.mcp_call.completed'`.
    pub r#type: String,
}

/// Emitted when an MCP tool call has failed.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseMcpCallFailed {
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.mcp_call.failed`
    pub r#type: String,
}

/// Emitted when an MCP tool call is in progress.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseMcpCallInProgress {
    /// The unique identifier of the MCP tool call item being processed.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.mcp_call.in_progress`
    pub r#type: String,
}

/// Emitted when the list of available MCP tools has been successfully retrieved.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseMcpListToolsCompleted {
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.mcp_list_tools.completed`
    pub r#type: String,
}

/// Emitted when the attempt to list available MCP tools has failed.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseMcpListToolsFailed {
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.mcp_list_tools.failed`
    pub r#type: String,
}

/// Emitted when the system is in the process of retrieving the list of available MCP tools.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseMcpListToolsInProgress {
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.mcp_list_tools.in_progress`
    pub r#type: String,
}

/// Emitted when a code interpreter call is in progress.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseCodeInterpreterCallInProgress {
    /// The unique identifier of the code interpreter tool call item.
    pub item_id: String,
    /// The index of the output item in the response for which the code interpreter call is in progress.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.code_interpreter_call.in_progress`
    pub r#type: String,
}

/// Emitted when the code interpreter is actively interpreting the code snippet.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseCodeInterpreterCallInterpreting {
    /// The unique identifier of the code interpreter tool call item.
    pub item_id: String,
    /// The index of the output item in the response for which the code interpreter is interpreting code.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.code_interpreter_call.interpreting`
    pub r#type: String,
}

/// Emitted when the code interpreter call is completed.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseCodeInterpreterCallCompleted {
    /// The unique identifier of the code interpreter tool call item.
    pub item_id: String,
    /// The index of the output item in the response for which the code interpreter call is completed.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.code_interpreter_call.completed`
    pub r#type: String,
}

/// Emitted when a partial code snippet is streamed by the code interpreter.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseCodeInterpreterCallCodeDelta {
    /// The partial code snippet being streamed by the code interpreter.
    pub delta: String,
    /// The unique identifier of the code interpreter tool call item.
    pub item_id: String,
    /// The index of the output item in the response for which the code is being streamed.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.code_interpreter_call_code.delta`
    pub r#type: String,
}

/// Emitted when the code snippet is finalized by the code interpreter.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseCodeInterpreterCallCodeDone {
    /// The partial code snippet being streamed by the code interpreter.
    pub code: String,
    /// The unique identifier of the code interpreter tool call item.
    pub item_id: String,
    /// The index of the output item in the response for which the code is finalized.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.code_interpreter_call_code.done`
    pub r#type: String,
}

/// Emitted when an annotation is added to output text content.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseOutputTextAnnotationAdded {
    /// The annotation object being added. (See annotation schema for details.)
    pub annotation: Annotation,
    /// The index of the annotation within the content part.
    pub annotation_index: i32,
    /// The index of the content part within the output item.
    pub content_index: i32,
    /// The unique identifier of the item to which the annotation is being added.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.output_text.annotation.added`.
    pub r#type: String,
}

/// Emitted when a response is queued and waiting to be processed.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseQueued {
    /// The full response object that is queued.
    pub response: Response,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.queued`.
    pub r#type: String,
}

/// Emitted when there is a delta (partial update) to the reasoning content.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseReasoningDelta {
    /// The index of the reasoning content part within the output item.
    pub content_index: i32,
    /// The partial update to the reasoning content.
    pub delta: ReasoningContentDelta,
    /// The unique identifier of the item for which reasoning is being updated.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `response.reasoning.delta`.
    pub r#type: String,
}

/// The partial update to the reasoning content.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ReasoningContentDelta {
    pub text: String,
}

/// Emitted when the reasoning content is finalized for an item.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseReasoningDone {
    /// The index of the reasoning content part within the output item.
    pub content_index: i32,
    /// The unique identifier of the item for which reasoning is finalized.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The finalized reasoning text.
    pub text: String,
    /// The type of the event. Always `response.reasoning.done`.
    pub r#type: String,
}

/// Emitted when there is a delta (partial update) to the reasoning summary content.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseReasoningSummaryDelta {
    /// The partial update to the reasoning summary content.
    pub delta: ReasoningContentDelta,
    /// The unique identifier of the item for which the reasoning summary is being updated.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The index of the summary part within the output item.
    pub summary_index: i32,
    /// The type of the event. Always `response.reasoning_summary.delta`.
    pub r#type: String,
}

/// Emitted when the reasoning summary content is finalized for an item.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ResponseReasoningSummaryDone {
    /// The unique identifier of the item for which the reasoning summary is finalized.
    pub item_id: String,
    /// The index of the output item in the response's output array.
    pub output_index: i32,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The index of the summary part within the output item.
    pub summary_index: i32,
    /// The finalized reasoning summary text.
    pub text: String,
    /// The type of the event. Always `response.reasoning_summary.done`.
    pub r#type: String,
}

/// Emitted when an error occurs.
#[derive(Clone, Serialize, Debug, Deserialize)]
pub struct ResponseError {
    pub error: ApiError,
    /// The sequence number of this event.
    pub sequence_number: i32,
    /// The type of the event. Always `error`.
    pub r#type: String,
}
