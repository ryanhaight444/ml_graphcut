const React = require("react")
const reactDOM = require("react-dom");

class Video extends React.Component{
    constructor(props){
        super(props);
    }
    componentDidUpdate(){
        if(this.props.vid_name){
            document.getElementById("left").load();
            document.getElementById("right").load();
        }
    }
    render(){
        let { vid_name } = this.props;

        if(!vid_name) return null;
        return(
            <div>
                <h2>Video: {vid_name}</h2>
                <video autoPlay loop muted id={`left`}>
                    <source src={`assets/${vid_name}-left.mp4`} type="video/mp4" />
                </video>
                &nbsp;
                <video autoPlay loop muted id="right">
                    <source src={`assets/${vid_name}-right.mp4`} type="video/mp4" />
                </video>
            </div>
        );
    }
}

class Main extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            vid_idx: -1,
        }

        this.video_names = ["brink", "fountain", "pigeons"];
        this.nextVid = this.nextVid.bind(this);
    }

    nextVid(){
        this.setState({vid_idx: this.state.vid_idx+1});
    }
    post(){
        return(
            <h1>Thanks! All done.</h1>
        );
    }

    render(){
        const { vid_idx } = this.state;
        const showVid = vid_idx > -1 && vid_idx < this.video_names.length;
        const vid_name = showVid ? this.video_names[vid_idx] : null;
        const tyMsg = vid_idx == this.video_names.length ? this.post() : null;
        return (
            <div>
                <a href="https://docs.google.com/forms/d/e/1FAIpQLSdOpq8Qh1JQwHXr4lsFwO2lGbcVC6b0OMQuHKTTwOAvzvWzbQ/viewform?usp=sf_link"
                target="_blank">
                    Click here to open the associated survey
                </a><br />
                <Video vid_name={vid_name} />
                <button onClick={this.nextVid}>Next Video</button>
                {tyMsg}
            </div>
        )
    }
}

reactDOM.render(
    <Main />,
    document.getElementById("container"),
);
